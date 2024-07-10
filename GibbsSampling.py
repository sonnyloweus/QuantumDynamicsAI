import torch
import torch.nn as nn
import random

class BatchedConditionalGibbsSampler(nn.Module):
    def __init__(self, batch_size, num_samples, mixing_time, joint_distribution):
        super().__init__()
        self.joint_distribution = joint_distribution  # this is the conditional distribution for the process, but
                                                      # joint in the gibbs sense as its a distribution over vectors
                                                      # and not individual bits
        
        if hasattr(joint_distribution, 'dim'):
            self.dim = self.joint_distribution.dim
        else:
            self.dim = self.joint_distribution.out_dim

        self.batch_size = batch_size
        self.mixing_time = mixing_time
        self.num_samples = num_samples

        # buffers allow for automatic movement onto the GPU
        self.register_buffer('initial_guess', (torch.rand(self.num_samples, self.batch_size, self.dim) < 0.25).float(), persistent=False)
        self.register_buffer('zeros', torch.zeros(self.num_samples, self.batch_size), persistent=False)
        self.register_buffer('ones', torch.ones(self.num_samples, self.batch_size), persistent=False)

    def estimate_conditional_expected_value(self, z, model_func=None):
        with torch.no_grad():
            w_samples = self.run_batched_gibbs(z)
            z_conditioned = z.expand(self.num_samples, -1, -1)
            return model_func(w_samples, z_conditioned).mean(dim=0)

    # batch size, num samples, dimension
    # w is of shape batch size x dimension
    @torch.no_grad()
    def run_batched_gibbs(self, z):
        x = self.initial_guess  # don’t ruin the initial guess TODO maybe we want to???
        conditioned = z.unsqueeze(0).expand(self.num_samples, -1, -1)  # sample size x batch size x cond dim
        for _ in range(self.mixing_time):
            x_prime = x.detach().clone()
            for j in range(self.dim):
                self.gibbs_update(x, x_prime, j, conditioned)  # sample size x batch size x dim
            x = x_prime
        self.initial_guess = x.detach().clone()  # initial guess changes throughout iterations
        return x

    @torch.no_grad()
    def gibbs_update(self, x, x_prime, index, conditioned):
        unnormalized_log_probs = None
        if hasattr(self.joint_distribution, 'unnormalized_log_probs_w_given_z_double_batched'):
            unnormalized_log_probs = lambda w: self.joint_distribution.unnormalized_log_probs_w_given_z_double_batched_params(conditioned, w, *self.joint_distribution.params())
        else:
            unnormalized_log_probs = lambda w: self.joint_distribution.unnormalized_log_probs_a_given_b_double_batched(w, conditioned)

        index_0 = torch.concat((x_prime[..., :index], x[..., index:]), dim=-1)
        index_1 = index_0.detach().clone()
        index_0[..., index] = self.zeros
        index_1[..., index] = self.ones
        log_likelihood_zero = unnormalized_log_probs(index_0)
        log_likelihood_one = unnormalized_log_probs(index_1)
        x_prime[..., index] = torch.distributions.bernoulli.Bernoulli(
            logits=log_likelihood_one - log_likelihood_zero).sample().squeeze(-1)


# Implements adapted gibbs sampling where we flip two bits instead of just one
class BatchedConditionalDoubleGibbsSampler(nn.Module):
    def __init__(self, batch_size, num_samples, mixing_time, joint_distribution, dim, num_ones):
        super().__init__()
        self.joint_distribution = joint_distribution  # this is the conditional distribution for the process, but
                                                      # joint in the gibbs sense as its a distribution over vectors
                                                      # and not individual bits

        self.dim = dim

        self.batch_size = batch_size
        self.mixing_time = mixing_time
        self.num_samples = num_samples
        self.num_ones = num_ones

        init = torch.zeros((self.num_samples, self.batch_size, self.dim))
        init[..., :num_ones] = 1
        # buffers allow for automatic movement onto the GPU
        self.register_buffer('initial_guess', init.float(), persistent=False)
        self.register_buffer('zeros', torch.zeros(self.num_samples, self.batch_size), persistent=False)
        self.register_buffer('ones', torch.ones(self.num_samples, self.batch_size), persistent=False)

    def estimate_conditional_expected_value(self, z, model_func=None):
        with torch.no_grad():
            w_samples = self.run_batched_gibbs(z)
            z_conditioned = z.expand(self.num_samples, -1, -1)
            return model_func(w_samples, z_conditioned).mean(dim=0)

    # batch size, num_samples, dimension
    # w is of shape batch size x dimension
    @torch.no_grad()
    def run_batched_gibbs(self, z):
        x = self.initial_guess
        conditioned = z.unsqueeze(0).expand(self.num_samples, -1, -1)  # sample size x batch size x cond dim
        for _ in range(self.mixing_time):
            x_prime = x.detach().clone()
            i, j = random.sample(range(self.dim), 2)
            self.gibbs_update(x, x_prime, i, j, conditioned)  # sample size x batch size x dim
            x = x_prime
        self.initial_guess = x.detach().clone()  # initial guess changes throughout iterations
        return x

    @torch.no_grad()
    def gibbs_update(self, x, x_prime, index_i, index_j, conditioned):
        unnormalized_log_probs = None
        if hasattr(self.joint_distribution, 'unnormalized_log_probs_w_given_z_double_batched'):
            unnormalized_log_probs = lambda w: self.joint_distribution.unnormalized_log_probs_w_given_z_double_batched_params(conditioned, w, *self.joint_distribution.params())
        else:
            unnormalized_log_probs = lambda w: self.joint_distribution.unnormalized_log_probs_a_given_b_double_batched(w, conditioned)

        index_0_0 = x.detach().clone()
        index_0_1 = x.detach().clone()
        index_1_0 = x.detach().clone()
        index_1_1 = x.detach().clone()

        index_0_0[..., index_i] = self.zeros
        index_0_0[..., index_j] = self.zeros

        index_0_1[..., index_i] = self.zeros
        index_0_1[..., index_j] = self.ones

        index_1_0[..., index_i] = self.ones
        index_1_0[..., index_j] = self.zeros

        index_1_1[..., index_i] = self.ones
        index_1_1[..., index_j] = self.ones

        log_likelihood_0_0 = unnormalized_log_probs(index_0_0)
        log_likelihood_0_1 = unnormalized_log_probs(index_0_1)
        log_likelihood_1_0 = unnormalized_log_probs(index_1_0)
        log_likelihood_1_1 = unnormalized_log_probs(index_1_1)

        logits = torch.cat([log_likelihood_0_0, log_likelihood_0_1, log_likelihood_1_0, log_likelihood_1_1], dim=2)
        random_sample = torch.distributions.categorical.Categorical(logits=logits).sample().squeeze(-1)
        index_i_value = torch.div(random_sample, 2, rounding_mode='trunc')
        index_j_value = random_sample % 2
        x_prime[..., [index_i, index_j]] = torch.dstack([index_i_value, index_j_value]).float()


if __name__ == '__main__':
##    from TrainDiscreteEmbeddingMIQuantumCircuit_v2 import EmbeddingMI2
##    model = EmbeddingMI2(4, 12, 6, num_ones=2).cpu()
##    model.load_state_dict(torch.load('/home/azureuser/cloudfiles/code/quantum_experiments_2/' + 'experiment_12_6_14999.model', map_location='cuda'))
##
##    encoder_sampler_double = BatchedConditionalDoubleGibbsSampler(batch_size=4, num_samples=1024, mixing_time=3, joint_distribution=model.decoder, dim=12, num_ones=2).cuda()
##
##    w = (torch.rand((4, 6)) > 0.5).float().cuda()
##    print(encoder_sampler_double.run_batched_gibbs(w).cpu().numpy())

    print("Sampling")
