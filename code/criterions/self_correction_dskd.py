import math
import torch
import torch.nn.functional as F
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA


class SelfCorrectionDSKD(DualSpaceKDWithCMA):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kl_orig_rate = getattr(args, 'kl_orig_rate', 1.0)
        self.original_model_path = getattr(args, 'original_model_path', None)
        self.original_model = None

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        # 1. Compute DSKD+DTW loss for (y*|x*) - corrected task
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        
        log = {}
        
        # Cross-entropy loss for corrected output
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        # Teacher forward pass for corrected task
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
        
        # DSKD loss for corrected task
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        # DTW loss for corrected task
        dtw_loss, log = self.compute_dtw_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        # 2. Compute KL divergence loss for (y1|x) - original task
        orig_outputs = model(
            input_data["original_input_ids"],
            attention_mask=input_data["original_attention_mask"],
            position_ids=input_data.get("original_position_ids", None), 
            output_hidden_states=False
        )
        
        # Get KL divergence loss for original task regularization
        kl_orig_loss, log = self.compute_kl_loss(
            orig_outputs, input_data, output_data, distiller, log
        )

        # 3. Combine all losses
        #    - The first three losses are token-sums over the corrected task,
        #      so we divide them by `batch_denom` to obtain a per-token average.
        #    - `kl_orig_loss` returned from `compute_kl_loss` is already a
        #      per-token average over the original task, so we do NOT divide it
        #      again.
        primary_loss = (
            self.ce_rate * ce_loss +
            self.kd_rate * kd_loss +
            self.dtw_rate * dtw_loss
        ) / batch_denom
        total_loss = primary_loss + self.kl_orig_rate * kl_orig_loss
        
        # Prepare logging: keep sums so that `record_logging_output` can still
        # normalise them by `batch_denom`.
        kl_orig_loss_sum = log.get("kl_orig_loss_sum", kl_orig_loss)
        log["loss"] = (
            self.ce_rate * ce_loss +
            self.kd_rate * kd_loss +
            self.dtw_rate * dtw_loss +
            self.kl_orig_rate * kl_orig_loss_sum
        )
        log["ce_loss"] = ce_loss
        log["kd_loss"] = kd_loss
        log["dtw_loss"] = dtw_loss
        log["kl_orig_loss"] = kl_orig_loss_sum   # raw sum for KL
        log["kl_orig_avg_loss"] = kl_orig_loss

        # Compute accuracy for corrected task
        accuracy = self.compute_token_accuracy(
            outputs.logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        # Compute accuracy for original task
        orig_accuracy = self.compute_token_accuracy(
            orig_outputs.logits, output_data["original_label"], 
        )
        log["original_accuracy"] = orig_accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return total_loss, logging_output

    def compute_kl_loss(self, orig_outputs, input_data, output_data, distiller, log):
        """
        Compute KL divergence loss:
        KL(p(y1_new|x) || p(y1|x)) where:
        - p(y1_new|x): current model's probability of predicting y1 given original prompt x
        - p(y1|x): original model's probability of predicting y1 given original prompt x
        
        This regularizes the model to maintain similar predictions for the original task.
        """
        
        # Get masks for original task
        orig_pad_mask = output_data["original_label"].ne(self.padding_id)
        orig_loss_mask = output_data["original_loss_mask"]
        
        # Current model's logits for predicting y1 given x (p(y1_new|x))
        current_logits = orig_outputs.logits
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        
        # Dynamically compute original model's predictions p(y1|x)
        if hasattr(distiller, 'original_model') and distiller.original_model is not None:
            # Option 1: Use original model to compute p(y1|x) dynamically
            with torch.no_grad():
                distiller.original_model.eval()
                original_outputs = distiller.original_model(
                    input_data["original_input_ids"],
                    attention_mask=input_data["original_attention_mask"],
                    position_ids=input_data.get("original_position_ids", None)
                )
                original_logits = original_outputs.logits
                original_probs = F.softmax(original_logits, dim=-1)
                log["using_dynamic_original_model"] = True
        elif "original_logits" in output_data and output_data["original_logits"] is not None:
            # Option 2: Use pre-computed original model logits if available
            original_logits = output_data["original_logits"]
            original_probs = F.softmax(original_logits, dim=-1)
            log["using_precomputed_original_logits"] = True
        else:
            # Option 3: Fallback - use temperature-scaled one-hot as surrogate for original model
            # This assumes the original model was very confident about the original response
            orig_targets = output_data["original_label"]
            temperature = 0.1  # Low temperature for high confidence
            
            # Create soft one-hot distribution
            one_hot = F.one_hot(orig_targets.clamp(min=0), num_classes=current_logits.size(-1)).float()
            # Apply temperature scaling to make it less sharp
            original_probs = F.softmax(one_hot / temperature, dim=-1)
            # Mask out padding tokens
            original_probs = original_probs * orig_pad_mask.unsqueeze(-1)
            log["using_surrogate_original_probs"] = True
        
        # Compute token-level KL divergence: KL(p(y1_new|x) || p(y1|x))
        kl_token = F.kl_div(
            current_log_probs,
            original_probs,
            reduction='none'
        ).sum(-1)  # sum over vocab dim, keep sequence length

        # Mask out padding / prompt tokens then compute
        # 1) sum over valid tokens (for logging)
        # 2) mean over valid tokens (for optimisation target)
        kl_token = kl_token * orig_loss_mask
        kl_sum = kl_token.sum()
        kl_mean = kl_sum / (orig_loss_mask.sum() + 1e-8)

        # Log sum and mean (token count no longer logged)
        log["kl_orig_loss_sum"] = kl_sum       # raw sum of KL over tokens
        log["avg_kl_orig_loss"] = kl_mean      # per-token mean KL

        return kl_mean, log

    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        """
        Override parent method to work with corrected task data
        This uses the same logic but with corrected input/output data
        """
        return super().compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

    def compute_dtw_loss(self, outputs, teacher_outputs, input_data, output_data, distiller, log):
        """
        Override parent method to work with corrected task data
        """
        return super().compute_dtw_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        ) 