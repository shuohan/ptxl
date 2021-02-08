class GANTrainer(Trainer):
    def __init__(self, generator, discriminator, pixel_criterion, adv_criterion,
                 generator_optimizer, discriminator_optimizer,
                 data_loader, pixel_lambda=0.9, adv_lambda=0.1,
                 num_epochs=500, num_batches=20):
        """Initialize
        
        """
        super().__init__(data_loader, num_epochs, num_batches)

        if self.use_gpu:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        self.models['generator'] = generator
        self.models['discriminator'] = discriminator

        self.pixel_criterion = pixel_criterion
        self.adv_criterion = adv_criterion
        self.pixel_lambda = pixel_lambda
        self.adv_lambda = adv_lambda
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.losses['gen_loss'] = Buffer(self.num_batches)
        self.losses['pixel_loss'] = Buffer(self.num_batches)
        self.losses['adv_loss'] = Buffer(self.num_batches)
        self.losses['dis_loss'] = Buffer(self.num_batches)

    def _train_on_batch(self, source, target):
        """Train the model for each batch
        
        Args:
            input (torch.Tensor): The input tensor to the model
            truth (torch.Tensor): The target/truth of the output of the model

        """
        if self.use_gpu:
            source = source.cuda()
            target = target.cuda()

        self.generator_optimizer.zero_grad()
        gen_pred = self.models['generator'](source)
        fake_pred = self.models['discriminator'](gen_pred, source)

        zeros = torch.zeros_like(fake_pred, requires_grad=False)
        ones = torch.ones_like(fake_pred, requires_grad=False)

        adv_loss = self.adv_criterion(fake_pred, ones)


class ConditionalGANValidator(Validator):
    """Validate the conditional GAN"""
    def _validate(self, input, truth):
        gen_pred = self.observable.models['generator'](input)
        fake_pred = self.observable.models['discriminator'](gen_pred, input)
        real_pred = self.observable.models['discriminator'](truth, input)

        zeros = torch.zeros_like(fake_pred, requires_grad=False)
        ones = torch.ones_like(fake_pred, requires_grad=False)

        adv_loss = self.observable.adv_criterion(fake_pred, ones).item()
        pixel_loss = self.observable.pixel_criterion(gen_pred, truth).item()
        gen_loss = self.observable.adv_lambda * adv_loss \
                 + self.observable.pixel_lambda * pixel_loss
        fake_loss = self.observable.adv_criterion(fake_pred, zeros).item()
        real_loss = self.observable.adv_criterion(real_pred, ones).item()
        dis_loss = 0.5 * (fake_loss + real_loss)

        self.losses['gen_loss'].append(gen_loss)
        self.losses['pixel_loss'].append(pixel_loss)
        self.losses['adv_loss'].append(adv_loss)
        self.losses['dis_loss'].append(dis_loss)

        self.evaluator.evaluate(gen_pred, truth)
