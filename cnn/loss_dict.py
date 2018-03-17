# Container to easily track and display multiple loss values for different
# decoder classes over time
class LossDict(object):
    def __init__(self, decoder_classes, domain_adversarial=False, gan=False, denoiser=False, phone=False):
        super().__init__()

        self.domain_adversarial = domain_adversarial
        self.gan = gan
        self.denoiser = denoiser
        self.phone = phone

        # Set up hierarchical loss dict
        self.decoder_class_losses = dict()
        self.elements_processed = {}
        for decoder_class in decoder_classes:
            self.elements_processed[decoder_class] = 0

            self.decoder_class_losses[decoder_class] = dict()
            self.elements_processed[decoder_class] = dict()

            if not self.denoiser or decoder_class == "clean":
                self.decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
                self.elements_processed[decoder_class]["autoencoding_recon_loss"] = 0

            if not self.denoiser or decoder_class == "dirty":
                self.decoder_class_losses[decoder_class]["transformation_recon_loss"] = 0.0
                self.elements_processed[decoder_class]["transformation_recon_loss"] = 0

            if self.domain_adversarial:
                self.decoder_class_losses[decoder_class]["domain_adversarial_loss"] = 0.0
                self.elements_processed[decoder_class]["domain_adversarial_loss"] = 0

            if self.gan:
                self.decoder_class_losses[decoder_class]["real_gan_loss"] = 0.0
                self.elements_processed[decoder_class]["real_gan_loss"] = 0
                self.decoder_class_losses[decoder_class]["fake_gan_loss"] = 0.0
                self.elements_processed[decoder_class]["fake_gan_loss"] = 0

            if self.phone:
                self.decoder_class_losses[decoder_class]["phone_loss"] = 0.0
                self.elements_processed[decoder_class]["phone_loss"] = 0

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
