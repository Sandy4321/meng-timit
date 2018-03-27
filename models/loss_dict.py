# Container to easily track and display multiple loss values for different
# decoder classes over time
class LossDict(object):
    def __init__(self, decoder_classes, model_type):
        super().__init__()

        allowed_model_types = ["acoustic_model", "enhancement_net", "end2end_phone_net", "multitask_net", "enhancement_md"]
        if model_type not in allowed_model_types:
            raise RuntimeError("Error: model type must be one of %s; got %s instead" % (str(allowed_model_types), model_type))
        self.model_type = model_type

        # Set up hierarchical loss dict
        self.decoder_class_losses = dict()
        self.elements_processed = {}
        for decoder_class in decoder_classes:
            self.elements_processed[decoder_class] = 0

            self.decoder_class_losses[decoder_class] = dict()
            self.elements_processed[decoder_class] = dict()

            if self.model_type in ["acoustic_model", "end2end_phone_net", "multitask_net"]:
                self.decoder_class_losses[decoder_class]["phones_xent"] = 0.0
                self.elements_processed[decoder_class]["phones_xent"] = 0

            if self.model_type in ["enhancement_net", "multitask_net"]:
                if decoder_class == "clean":
                    self.decoder_class_losses[decoder_class]["reconstruction_loss"] = 0.0
                    self.elements_processed[decoder_class]["reconstruction_loss"] = 0
                elif decoder_class == "dirty":
                    self.decoder_class_losses[decoder_class]["enhancement_loss"] = 0.0
                    self.elements_processed[decoder_class]["enhancement_loss"] = 0

            if self.model_type in ["enhancement_md"]:
                self.decoder_class_losses[decoder_class]["reconstruction_loss"] = 0.0
                self.elements_processed[decoder_class]["reconstruction_loss"] = 0
                self.decoder_class_losses[decoder_class]["transformation_loss"] = 0.0
                self.elements_processed[decoder_class]["transformation_loss"] = 0

    def __str__(self):
        output_str = "Losses:\n"
        for decoder_class in self.decoder_class_losses:
            output_str += "=> Class %s\n" % decoder_class
            class_loss = 0.0
            for loss_key in self.decoder_class_losses[decoder_class]:
                element_count = self.elements_processed[decoder_class][loss_key]
                if element_count > 0:
                    current_loss = self.decoder_class_losses[decoder_class][loss_key] / element_count
                else:
                    current_loss = 0.0
                class_loss += current_loss
                output_str += "===> %s: %.6f\n" % (loss_key, current_loss)
            output_str += "===> Total for class %s: %.6f\n" % (decoder_class, class_loss)
        output_str += "TOTAL: %.6f\n" % self.total_loss()
        return output_str

    def total_loss(self):
        loss = 0.0
        for decoder_class in self.decoder_class_losses:
            for loss_key in self.decoder_class_losses[decoder_class]:
                element_count = self.elements_processed[decoder_class][loss_key]
                if element_count > 0:
                    current_loss = self.decoder_class_losses[decoder_class][loss_key] / element_count
                else:
                    current_loss = 0.0
                loss += current_loss
        return loss

    def add(self, decoder_class, losses):
        for loss_key in losses:
            self.elements_processed[decoder_class][loss_key] += 1
            self.decoder_class_losses[decoder_class][loss_key] += losses[loss_key]
