import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from math import log

# class structure: all prediction models inherit the PredictionModel class.
# PredictionModel contains train_step(), valid_step(), test_step() and validation_epoch_end() methods.
#   PredictionModel also contains model performance metric outputs, tied into test_step.
# Individual models define loss functions and forward() methods.
# Data interfaces are defined in the script or in individual experiment class definitions.

# -------------------------------------------------
# -------------------------------------------------

# general prediction model class. No forward method or model initialization.
class PredictionModel(pl.LightningModule):
    
    def __init__(self,config):
        super(PredictionModel, self).__init__()
        
        self.lr         = config.learning_rate
        self.lr_factor  = config.learning_rate_factor

    def training_step(self, train_batch, batch_idx):
        src, trg = train_batch
        pred = self.forward(src,trg)
        loss, loss_dict = self.loss(trg,pred) # note: loss returns total loss objective and dict of summands
        self.logger.experiment.log({'train_loss': loss})
        for k in loss_dict.keys():
            self.logger.experiment.log({f'train_{k}': loss_dict[k]})
        return {'loss': loss}
    
    def validation_step(self, valid_batch, batch_idx):
        src, trg = valid_batch
        pred = self.forward(src,trg)
        loss, loss_dict = self.loss(trg,pred)
        self.logger.experiment.log({'valid_loss': loss})
        for k in loss_dict.keys():
            self.logger.experiment.log({f'valid_{k}': loss_dict[k]})
        return {'valid_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        self.logger.experiment.log({'avg_valid_loss': avg_loss})
        return {'avg_valid_loss': avg_loss}

    def test_step(self, test_batch, batch_idx):
        src, trg = test_batch
        pred = self.forward(src,trg)
        loss, loss_dict = self.loss(trg,pred)
        self.logger.experiment.log({'test_loss': loss})
        for k in loss_dict.keys():
            self.logger.experiment.log({f'test_{k}': loss_dict[k]})
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),self.lr) # set rate?
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=self.lr_factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'avg_valid_loss'
        }

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        None

# -------------------------------------------------
# -------------------------------------------------

# vanilla LFADS model - controller implemented, removed if controller size is 0
class Lfads(PredictionModel):
    
    def __init__(self,config):
        super(Lfads, self).__init__(config)

        # model parameters
        # self.input_size             = None # defined by dataset
        self.g_encoder_size         = config.g_encoder_size
        self.c_encoder_size         = config.c_encoder_size
        self.g_latent_size          = config.g_latent_size
        self.u_latent_size          = config.u_latent_size
        self.controller_size        = config.controller_size
        self.generator_size         = config.generator_size
        self.factor_size            = config.factor_size

        # model regularization
        self.clip_val               = config.clip_val
        self.max_norm               = config.max_norm
        self.do_normalize_factors   = config.do_normalize_factors
        self.factor_bias            = config.factor_bias # bool

        # loss parameters
        self.loss_weight_dict       = config.loss_weight_dict
        # self.l2_gen_scale           = config.l2_gen_scale
        # self.l2_con_scale           = config.l2_con_scale

        self.dropout                = config.dropout

        # encoder RNN
        self.encoder    = LFADS_Encoder(
            input_size      = self.input_size,
            g_encoder_size  = self.g_encoder_size,
            c_encoder_size  = self.c_encoder_size,
            g_latent_size   = self.g_latent_size,
            clip_val        = self.clip_val,
            dropout         = self.dropout
            )

        # controller RNN
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.controller  = LFADS_ControllerCell(input_size      = self.c_encoder_size*2 + self.factor_size,
                                                    controller_size = self.controller_size,
                                                    u_latent_size   = self.u_latent_size,
                                                    clip_val        = self.clip_val,
                                                    dropout         = self.dropout)
        
        # generator RNN
        self.generator   = LFADS_GeneratorCell(input_size     = self.u_latent_size,
                                               generator_size = self.generator_size,
                                               factor_size    = self.factor_size,
                                               clip_val       = self.clip_val,
                                               factor_bias    = self.factor_bias,
                                               dropout        = self.dropout)

        # dense layers
        if self.g_latent_size == self.generator_size:
            self.fc_genstate = nn.Identity(in_features=self.g_latent_size, out_features=self.generator_size)
        else:
            self.fc_genstate = nn.Linear(in_features=self.g_latent_size, out_features=self.generator_size)
        
        # learnable biases
        self.g_encoder_init = nn.Parameter(torch.zeros(2, self.g_encoder_size))
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c_encoder_size))
            self.controller_init = nn.Parameter(torch.zeros(self.controller_size))

        # Initialize priors
        self.g_prior_mean = torch.ones(self.g_latent_size) * self.prior['g0']['mean']['value']
        if self.prior['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g_latent_size) * log(self.prior['g0']['var']['value'])
        if self.prior['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.u_prior_gp_mean = torch.ones(self.u_latent_size) * self.prior['u']['mean']['value']
            if self.prior['u']['mean']['learnable']:
                self.u_prior_gp_mean = nn.Parameter(self.u_prior_gp_mean)
            self.u_prior_gp_logvar = torch.ones(self.u_latent_size) * log(self.prior['u']['var']['value'])
            if self.prior['u']['var']['learnable']:
                self.u_prior_gp_logvar = nn.Parameter(self.u_prior_gp_logvar)
            self.u_prior_gp_logtau = torch.ones(self.u_latent_size) * log(self.prior['u']['tau']['value'])
            if self.prior['u']['tau']['learnable']:
                self.u_prior_gp_logtau = nn.Parameter(self.u_prior_gp_logtau)

        # rate estimates
        self.fc_out = nn.Linear(in_features=self.factor_size, out_features=self.input_size)

        # create model loss function
        self.objective = LFADS_Loss(
            loss_weight_dict = self.loss_weight_dict,
            l2_con_scale = self.loss_weight_dict['l2_con_scale'],
            l2_gen_scale = self.loss_weight_dict['l2_gen_scale']
        )
        
        self.initialize_weights()
    
    def forward(self,src,trg):

        # initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(src)

        # encoder outputs (distribution IC)
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(src,(g_encoder_state, c_encoder_state))

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state (why is it done this way? All separate layers should be visible?)
        factor_state = self.generator.fc_factors(self.generator.dropout(generator_state))
        
        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)

        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        # linear reconstruction from factors
        recon = {'rates': self.fc_out(factors)}
        recon['data'] = recon['rates'].clone()

        return (recon, factors, gen_inputs)

    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        # Check dimensions
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'Input is expected to have dimensions [%i, %i, %i]'%(self.steps_size, self.batch_size, self.input_size)
        
        g_encoder_state  = (torch.ones(self.batch_size, 2,  self.g_encoder_size, device=self.device) * self.g_encoder_init).permute(1, 0, 2)
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            c_encoder_state  = (torch.ones(self.batch_size, 2,  self.c_encoder_size, device=self.device) * self.c_encoder_init).permute(1, 0, 2)
            controller_state = torch.ones(self.batch_size, self.controller_size, device=self.device) * self.controller_init
            return g_encoder_state, c_encoder_state, controller_state
        else:
            return g_encoder_state, None, None
    
    def _gp_to_normal(self, gp_mean, gp_logvar, gp_logtau, process):
        '''
        _gp_to_normal(gp_mean, gp_logvar, gp_logtau, process)
        
        Convert gaussian process with given process mean, process log-variance, process tau, and realized process
        to mean and log-variance of diagonal Gaussian for each time-step
        '''
        
        mean   = gp_mean * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        logvar = gp_logvar * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        
        mean   = torch.cat((mean, gp_mean + (process[:-1] - gp_mean) * torch.exp(-1/gp_logtau.exp())))
        logvar = torch.cat((logvar, torch.log(1 - torch.exp(-1/gp_logtau.exp()).pow(2)) + gp_logvar * torch.ones(process.shape[0]-1, process.shape[1], process.shape[2], device=self.device)))
        return mean.permute(1, 0, 2), logvar.permute(1, 0, 2)
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
        
        def standard_init(weights):
            k = weights.shape[1] # dimensionality of inputs
            weights.data.normal_(std=k**-0.5) # inplace resetting W ~ N(0, 1/sqrt(K))
        
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    standard_init(p)

            if self.do_normalize_factors:
                self.normalize_factors()
                
    def normalize_factors(self):
        self.generator.fc_factors.weight.data = F.normalize(self.generator.fc_factors.weight.data, dim=1)
        
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler
    
    def kl_div(self):
        kl = kldiv_gaussian_gaussian(post_mu  = self.g_posterior_mean,
                                     post_lv  = self.g_posterior_logvar,
                                     prior_mu = self.g_prior_mean,
                                     prior_lv = self.g_prior_logvar)
        if self.u_latent_size > 0:
            kl += kldiv_gaussian_gaussian(post_mu  = self.u_posterior_mean,
                                          post_lv  = self.u_posterior_logvar,
                                          prior_mu = self.u_prior_mean,
                                          prior_lv = self.u_prior_logvar)
        return kl

    def loss(self,trg,pred):
        loss, loss_dict = self.objective(trg,pred,self) # the multi-objective loss object needs to see the model loss methods.
        # hopefully this doesn't create some kind of broken loop. :/
        return loss, loss_dict

# -------------------------------------------------
# -------------------------------------------------

# seq2seq model. Nothing too fancy here yet!
class Seq2Seq(PredictionModel):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        # create model: encoder, decoder, fc output, attention (maybe?)

    def forward(self,src,trg):
        return None

    def loss(self,trg,pred):
        return None, None

# -------------------------------------------------
# LFADS modules
# -------------------------------------------------

class LFADS_Encoder(nn.Module):
    '''
    LFADS_Encoder
    
    LFADS Encoder Network 
    
    __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0)
    
    Required Arguments:
        - input_size (int):  size of input dimensions
        - g_encoder_size (int):  size of generator encoder network
        - g_latent_size (int): size of generator ic latent variable
        
    Optional Arguments:
        - c_encoder_size (int): size of controller encoder network
        - dropout (float): dropout probability
        - clip_val (float): RNN hidden state value limit
        
    '''
    def __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0):
        super(LFADS_Encoder, self).__init__()
        self.input_size      = input_size
        self.g_encoder_size  = g_encoder_size
        self.c_encoder_size  = c_encoder_size
        self.g_latent_size   = g_latent_size
        self.clip_val        = clip_val

        self.dropout = nn.Dropout(dropout)
        
        # g Encoder BiRNN
        self.gru_g_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.g_encoder_size, bidirectional=True)
        # g Linear mapping
        self.fc_g0_theta    = nn.Linear(in_features= 2 * self.g_encoder_size, out_features= self.g_latent_size * 2)
        
        if self.c_encoder_size > 0:
            # c encoder BiRNN
            self.gru_c_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.c_encoder_size, bidirectional=True)
            
    def forward(self, input, hidden):
        self.gru_g_encoder.flatten_parameters()
        if self.c_encoder_size > 0:
            self.gru_c_encoder.flatten_parameters()
        gru_g_encoder_init, gru_c_encoder_init = hidden
        
        # Run bidirectional RNN over data
        out_gru_g_enc, hidden_gru_g_enc = self.gru_g_encoder(self.dropout(input), gru_g_encoder_init.contiguous())
        hidden_gru_g_enc = self.dropout(hidden_gru_g_enc.clamp(min=-self.clip_val, max=self.clip_val))
        hidden_gru_g_enc = torch.cat((hidden_gru_g_enc[0], hidden_gru_g_enc[1]), dim=1)
        
        g0_mean, g0_logvar = torch.split(self.fc_g0_theta(hidden_gru_g_enc), self.g_latent_size, dim=1)
        
        if self.c_encoder_size > 0:
            out_gru_c_enc, hidden_gru_c_enc = self.gru_c_encoder(self.dropout(input), gru_c_encoder_init.contiguous())
            out_gru_c_enc = out_gru_c_enc.clamp(min=-self.clip_val, max=self.clip_val)
        
            return g0_mean, g0_logvar, out_gru_g_enc, out_gru_c_enc
        
        else:
            
            return g0_mean, g0_logvar, out_gru_g_enc, None
        
class LFADS_ControllerCell(nn.Module):
    
    def __init__(self, input_size, controller_size, u_latent_size, dropout = 0.0, clip_val=5.0):
        super(LFADS_ControllerCell, self).__init__()
        self.input_size      = input_size
        self.controller_size = controller_size
        self.u_latent_size   = u_latent_size
        self.clip_val        = clip_val
        
        self.dropout = nn.Dropout(dropout)
        
        self.gru_controller  = LFADS_GenGRUCell(input_size  = self.input_size, hidden_size = self.controller_size)
        self.fc_u_theta = nn.Linear(in_features = self.controller_size, out_features=self.u_latent_size * 2)
        
    def forward(self, input, hidden):
        controller_state = hidden
        controller_state = self.gru_controller(self.dropout(input), controller_state)
        controller_state = controller_state.clamp(-self.clip_val, self.clip_val)
        u_mean, u_logvar = torch.split(self.fc_u_theta(controller_state), self.u_latent_size, dim=1)
        return u_mean, u_logvar, controller_state
    
class LFADS_GeneratorCell(nn.Module):
    
    def __init__(self, input_size, generator_size, factor_size, attention=False, dropout = 0.0, clip_val = 5.0, factor_bias = False):
        super(LFADS_GeneratorCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size
        
        self.dropout = nn.Dropout(dropout)
        self.clip_val = clip_val
        
        self.gru_generator = LFADS_GenGRUCell(input_size=input_size, hidden_size=generator_size)
        self.fc_factors = nn.Linear(in_features=generator_size, out_features=factor_size, bias=factor_bias)
        if attention:
            self.attention = nn.F
        
    def forward(self, input, hidden):
        
        generator_state = hidden
        generator_state = self.gru_generator(input, generator_state)
        generator_state = generator_state.clamp(min=-self.clip_val, max=self.clip_val)
        factor_state    = self.fc_factors(self.dropout(generator_state))
        
        return generator_state, factor_state

# -------------------------------------------------
# seq2seq modules
# -------------------------------------------------

class Seq2Seq_Encoder(nn.Module):
    
    def __init__(self,config):
        None
    
    def forward(self,config):
        None

class Seq2Seq_Decoder(nn.Module):
    
    def __init__(self,config):
        None

    def forward(self,config):
        None

# -------------------------------------------------
# general modules
# -------------------------------------------------

class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

# -------------------------------------------------
# LFADS objective
# -------------------------------------------------

class Base_Loss(nn.Module):
    def __init__(self, loss_weight_dict, l2_gen_scale=0.0, l2_con_scale=0.0):
        super(Base_Loss, self).__init__()
        self.loss_weights = loss_weight_dict
        self.l2_gen_scale = l2_gen_scale
        self.l2_con_scale = l2_con_scale
        
    def forward(self, x_orig, x_recon, model):
        pass

    def weight_schedule_fn(self, step):
        '''
        weight_schedule_fn(step)
        
        Calculate the KL and L2 regularization weights from the current training step number. Imposes
        linearly increasing schedule on regularization weights to prevent early pathological minimization
        of KL divergence and L2 norm before sufficient data reconstruction improvement. See bullet-point
        4 of section 1.9 in online methods
        
        required arguments:
            - step (int) : training step number
        '''
        
        for key in self.loss_weights.keys():
            # Get step number of scheduler
            weight_step = max(step - self.loss_weights[key]['schedule_start'], 0)
            
            # Calculate schedule weight
            self.loss_weights[key]['weight'] = max(min(self.loss_weights[key]['max'] * weight_step/ self.loss_weights[key]['schedule_dur'], self.loss_weights[key]['max']), self.loss_weights[key]['min'])

    def any_zero_weights(self):
        for key, val in self.loss_weights.items():
            if val['weight'] == 0:
                return True
            else:
                pass
        return False

class LFADS_Loss(Base_Loss):
    def __init__(self,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(LFADS_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = LogLikelihoodGaussian()
        
    # note: when called as a module in LfadsModel, it will be self.LfadsLoss(trg,pred,self)
    # 
    # @torch.jit.script   # does this break anything?
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        recon_loss = -self.loglikelihood(x_orig, x_recon[0]['data'])    # x_recon = (recon, factors, g_input)

        # access model methods/loss terms instead of DataParallel methods - may not be necessary in ptl
        # if type(model) is 'DataParallel':
        #     model = model.module

        kl_loss = kl_weight * model.kl_div()
    
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model, 'controller'):
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_loss +  kl_loss + l2_loss
        loss_dict = {'recon' : float(recon_loss.data),
                     'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}

        # if torch.isinf(loss):
        #     import matplotlib.pyplot as plt
        #     breakpoint()

        return loss, loss_dict

class LFADS_GRUCell(nn.Module):
    
    '''
    LFADS_GRUCell class. Implements the Gated Recurrent Unit (GRU) used in LFADS Encoders. More obvious
    relation to the equations (see https://en.wikipedia.org/wiki/Gated_recurrent_unit), along with
    a hack to help learning
    
    __init__(self, input_size, hidden_size, forget_bias=1.0)
    
    required arguments:
     - input_size (int) : size of inputs
     - hidden_size (int) : size of hidden state
     
    optional arguments:
     - forget_bias (float) : hack to help learning, added to update gate in sigmoid
    '''
    
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(LFADS_GRUCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        
        # Concatenated sizes
        self._xh_size = input_size + hidden_size
        self._ru_size = hidden_size * 2
        
        # r, u = W([x, h]) + b
        self.fc_xh_ru = nn.Linear(in_features= self._xh_size, out_features= self._ru_size)
        # c = W([x, h*r]) + b
        self.fc_xhr_c = nn.Linear(in_features= self._xh_size, out_features= self.hidden_size)
        
    def forward(self, x, h):
        '''
        Forward method - Gated Recurrent Unit forward pass with forget bias
        
        forward(self, x, h):
        
        required arguments:
          - x (torch.Tensor) : GRU input
          - h (torch.Tensor) : GRU hidden state
        
        returns
          - h_new (torch.Tensor) : updated GRU hidden state
        '''
        
        # Concatenate input and hidden state
        xh  = torch.cat([x, h], dim=1)
        
        # Compute reset gate and update gate vector
        r,u = torch.split(self.fc_xh_ru(xh),
                          split_size_or_sections=self.hidden_size,
                          dim = 1)
        r,u = torch.sigmoid(r), torch.sigmoid(u + self.forget_bias)
        
        # Concatenate input and hadamard product of hidden state and reset gate
        xrh = torch.cat([x, torch.mul(r, h)], dim=1)
        
        # Compute candidate hidden state
        c   = torch.tanh(self.fc_xhr_c(xrh))
        
        # Return new hidden state as a function of update gate, current hidden state, and candidate hidden state
        return torch.mul(u, h) + torch.mul(1 - u, c)
    
class LFADS_GenGRUCell(nn.Module):
    '''
    LFADS_GenGRUCell class. Implements gated recurrent unit used in LFADS generator and controller. Same as
    LFADS_GRUCell, but parameters transforming hidden state are kept separate for computing L2 cost (see 
    bullet point 2 of section 1.9 in online methods). Also does not create parameters transforming inputs if 
    no inputs exist.
    
    __init__(self, input_size, hidden_size, forget_bias=1.0)
    '''
    
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(LFADS_GenGRUCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        
        # Concatenated size
        self._ru_size    = self.hidden_size * 2
        
        # Create parameters for transforming inputs if inputs exist
        if self.input_size > 0:
            
            # rx ,ux = W(x) (No bias in tensorflow implementation)
            self.fc_x_ru = nn.Linear(in_features= self.input_size, out_features= self._ru_size, bias=False)
            # cx = W(x) (No bias in tensorflow implementation)
            self.fc_x_c  = nn.Linear(in_features= self.input_size, out_features= self.hidden_size, bias=False)
        
        # Create parameters transforming hidden state
        
        # rh, uh = W(h) + b
        self.fc_h_ru = nn.Linear(in_features= self.hidden_size, out_features= self._ru_size)
        # ch = W(h) + b
        self.fc_rh_c = nn.Linear(in_features=self.hidden_size, out_features= self.hidden_size)
        
    def forward(self, x, h):
        '''
        Forward method - Gated Recurrent Unit forward pass with forget bias, weight on inputs and hidden state kept separate.
        
        forward(self, x, h):
        
        required arguments:
          - x (torch.Tensor) : GRU input
          - h (torch.Tensor) : GRU hidden state
        
        returns
          - h_new (torch.Tensor) : updated GRU hidden state
        '''
        
        # Calculate reset and update gates from input
        if self.input_size > 0 and x is not None:
            r_x, u_x = torch.split(self.fc_x_ru(x),
                                   split_size_or_sections=self.hidden_size,
                                   dim = 1)
        else:
            r_x = 0
            u_x = 0
        
        # Calculate reset and update gates from hidden state
        r_h, u_h = torch.split(self.fc_h_ru(h),
                               split_size_or_sections=self.hidden_size,
                               dim = 1)
        
        # Combine reset and updates gates from hidden state and input
        r = torch.sigmoid(r_x + r_h)
        u = torch.sigmoid(u_x + u_h + self.forget_bias)
        
        # Calculate candidate hidden state from input
        if self.input_size > 0 and x is not None:
            c_x = self.fc_x_c(x)
        else:
            c_x = 0
        
        # Calculate candidate hidden state from hadamard product of hidden state and reset gate
        c_rh = self.fc_rh_c(r * h)
        
        # Combine candidate hidden state vectors
        c = torch.tanh(c_x + c_rh)
        
        # Return new hidden state as a function of update gate, current hidden state, and candidate hidden state
        return u * h + (1 - u) * c
    
    def hidden_weight_l2_norm(self):
        return self.fc_h_ru.weight.norm(2).pow(2)/self.fc_h_ru.weight.numel() + self.fc_rh_c.weight.norm(2).pow(2)/self.fc_rh_c.weight.numel()

# objective support functions
class LogLikelihoodGaussian(nn.Module):
    def __init__(self, mse=True):
        super(LogLikelihoodGaussian, self).__init__()
        self.mse = mse
        
    def forward(self, x, mean, logvar=None):
        if logvar is not None:
            out = loglikelihood_gaussian(x, mean, logvar)
        else:
            if self.mse:
                out = -torch.nn.functional.mse_loss(x, mean, reduction='sum')/x.shape[0]
            else:
                out = -torch.nn.functional.l1_loss(x,mean,reduction='sum')/x.shape[0]
        return out
    
def loglikelihood_gaussian(x, mean, logvar):
    from math import pi
    return -0.5*(log(2*pi) + logvar + ((x - mean).pow(2)/torch.exp(logvar))).mean(dim=0).sum()
        

def kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv)

    KL-Divergence between a prior and posterior diagonal Gaussian distribution.

    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).mean(dim=0).sum()
    return klc
