from prediction import PredictionModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# -------------------------------------------------

class ConvAE(PredictionModel):

    def __init__(self, input_size, latent_size, src_len, trg_len, n_kernels, kernel_size, pool_size=2, dropout=0.3, learning_rate=1e-3):
        super(ConvAE, self).__init__(input_size, learning_rate, 0.5)
        self.input_size     = input_size
        self.latent_size    = latent_size
        self.src_len        = src_len 
        self.trg_len        = trg_len
        self.n_kernels      = n_kernels 
        self.kernel_size    = kernel_size
        self.pool_size      = pool_size
        self.dropout        = dropout
        self.learning_rate  = learning_rate

        self.flat_size      = src_len // pool_size * n_kernels

        self.encoder_1  = nn.Sequential(
            # Block 1 (time conv)
            nn.Conv2d(
                in_channels     = 1,
                out_channels    = self.n_kernels,
                kernel_size     = (self.kernel_size, 1),
                stride          = 1,
                padding         = (self.kernel_size//2,0), # is this right?
                padding_mode    = 'replicate'
            ),
            nn.BatchNorm2d(self.n_kernels),
            nn.ELU(),
            nn.Dropout(self.dropout)
        )
        self.encoder_2  = nn.Sequential(
            # Block 2 (channel conv)
            nn.Conv2d(
                in_channels     = self.n_kernels,
                out_channels    = self.n_kernels,
                kernel_size     = (1, self.input_size),
                stride          = 1
            ),
            nn.BatchNorm2d(self.n_kernels),
            nn.ELU(),
            nn.Dropout(self.dropout)
        )
        self.encoder_3  = nn.Sequential(
            # Block 3 (reduce, flatten)
            nn.AvgPool2d(
                kernel_size     = (2,1),
                stride          = 2
            ),
            nn.Flatten()
        )

        self.l_mean     = nn.Linear(
            in_features     = self.flat_size,
            out_features    = self.latent_size
        )

        self.l_logvar   = nn.Linear(
            in_features     = self.flat_size,
            out_features    = self.latent_size
        )

        self.decoder_1  = nn.Sequential(
            # Block 1 (reform)
            nn.Linear(
                in_features     = self.latent_size,
                out_features    = self.flat_size
            )
        ) # compose with tensor view
        self.decoder_2  = nn.Sequential(
            # Block 2 (deconv channels)
            nn.Upsample(
                size = (trg_len, 1)
            ),
            nn.ConvTranspose2d(
                in_channels     = self.n_kernels,
                out_channels    = self.n_kernels,
                kernel_size     = (1, self.input_size),
                stride          = 1
            ),
            nn.BatchNorm2d(self.n_kernels),
            nn.ELU(),
            nn.Dropout(self.dropout),
        )
        self.decoder_3  = nn.Sequential(
            # Block 3 (deconv time)
            nn.ConvTranspose2d(
                in_channels     = self.n_kernels,
                out_channels    = 1,
                kernel_size     = (self.kernel_size, 1),
                stride          = 1,
                padding         = (self.kernel_size//2, 0),
                padding_mode    = 'zeros'
            ),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def forward(self, src, trg):
        n_batch, src_len, src_ch    = src.shape
        _, trg_len, trg_ch          = trg.shape
        # shape size check?
        enc         = self.encoder_1(src.unsqueeze(1))
        enc         = self.encoder_2(enc)
        enc         = self.encoder_3(enc)
        z_sample    = self._sample_gaussian(self.l_mean(enc),self.l_logvar(enc))
        dec         = self.decoder_1(z_sample).view(n_batch,self.n_kernels,self.src_len//self.pool_size,1)
        dec         = self.decoder_2(dec)
        dec         = self.decoder_3(dec)
        return dec.squeeze(1)

    def loss(self, pred, trg):
        loss = F.mse_loss(pred,trg,reduction='mean')
        loss_dict = {
            'pred_mse': loss
        }
        return (loss, loss_dict)
    
    @staticmethod
    def _sample_gaussian(mean,logvar):
        eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).type_as(mean)
        return torch.exp(logvar*0.5)*eps + mean