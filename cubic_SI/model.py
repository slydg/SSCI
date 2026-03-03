import torch
import numpy as np
import copy
import cubic_SI.networks
import os



import cubic_SI.computations
import cubic_SI.model_train



class Cubic_SI_model(object):
    def __init__(self,data_tensor,timepoints,dataloader_input=False,conditional=True,condition_tensor=None,N_training=1000,B=128,steps=60,func_type='None',u_t=lambda x: 1, model_lr=1e-3,hiden_size=128,n_layers=4,spline=True,data_transformer=None,
    early_stop=False,patience=10,decay=0.8,lambda_=1e-3,save=True,plot_loss=True,save_path='model_history',record_gap=1,use_mlp=True,use_conditional_path_encoder=False,hist_times=None,C_d=None,dynamic_conditions=False,bounded_out=False,d=128,concentrate=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_tensor=data_tensor
        self.timepoints = timepoints
        self.condition_tensor=condition_tensor
        self.N_training=N_training
        self.B=B
        self.dataloader_input=dataloader_input
        self.d=data_tensor.shape[-1] if not dataloader_input else d
        self.conditional = conditional
        if C_d is not None:
            self.C_d = C_d
        elif self.conditional:
            self.C_d = condition_tensor.shape[-1] if not dataloader_input else C_d
        else:
            self.C_d = None
        self.steps=steps
        self.func_type = func_type
        self.u_t=u_t
        self.spline = spline
        self.model_lr=model_lr
        self.limit=1e-8
        self.data_transformer = data_transformer
        
        self.delta_t = (timepoints[-1] - timepoints[0])/self.steps
        
        self.early_stop = early_stop
        self.patience = patience
        self.decay=decay
        
        self.lambda_ = lambda_
        self.t_list=list(np.arange(timepoints[0],timepoints[-1],self.delta_t)) + [timepoints[-1]]
        self.save = save
        self.plot_loss = plot_loss
        self.save_path = save_path
        self.record_gap = record_gap
        self.dynamic_conditions=dynamic_conditions

        os.makedirs(self.save_path, exist_ok=True)


        self.t_lists_stage=[[] for i in range(len(self.timepoints)-1)]
        for t in self.t_list:
            for t_index,t_point in enumerate(self.timepoints[1:]):
                if t >= t_point:
                    continue
                else:
                    self.t_lists_stage[t_index].append(t)
                    break
        for stage,t_lst in enumerate(self.t_lists_stage):
            t_lst.append(float(self.timepoints[stage+1]))
        if not use_mlp:
            self.b_m = cubic_SI.networks.UNetWithLinear(x_size=self.d,t_size=hiden_size,output_size=self.d,hidden_size=hiden_size,n_layers=n_layers,condition_input_size=None if not self.conditional else self.C_d,bounded_out=bounded_out,concentrate=concentrate).to(self.device)
            self.kappa_m = cubic_SI.networks.UNetWithLinear(x_size=self.d,t_size=hiden_size,output_size=self.d,hidden_size=hiden_size,n_layers=n_layers,condition_input_size=None if not self.conditional else self.C_d,concentrate=concentrate).to(self.device)
        else:
            self.b_m = cubic_SI.networks.MLPNet_Fair(data_dim=self.d,cond_dim=self.C_d,latent_dim=hiden_size,vae_hidden_dim=hiden_size,dyn_hidden_dim=hiden_size,bounded_out=bounded_out).to(self.device)
            self.kappa_m = cubic_SI.networks.MLPNet_Fair(data_dim=self.d,cond_dim=self.C_d,latent_dim=hiden_size,vae_hidden_dim=hiden_size,dyn_hidden_dim=hiden_size).to(self.device)

        self.use_conditional_path_encoder = use_conditional_path_encoder
        self.hist_times = hist_times
        self.conditional_path_encoder = None
        if self.use_conditional_path_encoder:
            self.conditional_path_encoder = cubic_SI.networks.MotionTransformerEncoder(input_dim=self.d,latent_dim=self.C_d).to(self.device)


        self.model_trainer = cubic_SI.model_train.model_trainer(
            self.model_lr,self.N_training,self.B,
            self.timepoints,self.t_list,
            cubic_SI.computations.gamma_function,
            cubic_SI.computations.gamma_derivative,
            self.func_type,
            spline=self.spline,linear=not self.spline,
            decay=self.decay,
            early_stop=self.early_stop,patience=self.patience,
            plot_loss=self.plot_loss,save_model=self.save,
            save_path=self.save_path,
            record_gap=self.record_gap,
            conditional=self.conditional,
            conditional_path_encoder=self.conditional_path_encoder,
            hist_times=self.hist_times,
            dynamic_conditions=self.dynamic_conditions,
            data_transformer = self.data_transformer
        )


    def model_load(self,model_path='model.pt'):
        models_dict=torch.load(model_path)
        self.b_m.load_state_dict(models_dict['b_m'])
        self.kappa_m.load_state_dict(models_dict['kappa_m'])



    def train(self):
        if not self.dataloader_input:
            if self.condition_tensor is None:
                dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.data_tensor), batch_size=self.B, num_workers=0, shuffle=True)
            else:
                dataloader = torch.utils.data.DataLoader(
                                                        torch.utils.data.TensorDataset(self.data_tensor,  self.condition_tensor),
                                                        batch_size=self.B,
                                                        num_workers=0,
                                                        shuffle=True
                                                    )
        else:
            dataloader = self.data_tensor
        self.model_trainer.train(self.b_m,self.kappa_m,dataloader)


    def eval(self,test_0,conditions=None,SDE=True):
        self.b_m.eval()
        self.kappa_m.eval()
        with torch.no_grad():
            x_f=self.forward_generate(test_0,conditions=conditions,SDE=SDE)
        return x_f

    def forward_generate(self,x_0,conditions=None,SDE=True):
        def gamma(t):
            return cubic_SI.computations.gamma_function(t,torch.Tensor(self.timepoints).to(self.device),self.func_type)
        return cubic_SI.computations.generate_path(self.t_lists_stage,x_0,self.b_m,self.kappa_m,self.delta_t,self.u_t,gamma,SDE=SDE,conditions=conditions,conditional_path_encoder=self.conditional_path_encoder,hist_times=self.hist_times,dynamic_conditions=self.dynamic_conditions)
        
