import torch
import torch.nn
import torch.nn as nn
import math
import numpy as np
rng = np.random.default_rng()

class EnergyBasedModelEmbeddingDynamics(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim # dimension of y
        if hidden_dim is None:
            hidden_dim = 32

        self.linear_1_weight = nn.Parameter(torch.zeros((hidden_dim, self.dim**2)))
        self.linear_1_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_2_weight = nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.linear_2_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_3_weight = nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.linear_3_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_4_weight = nn.Parameter(torch.zeros((1, hidden_dim)))
        self.linear_4_bias = nn.Parameter(torch.zeros((1)))

        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.linear_1_weight)
        torch.nn.init.xavier_uniform_(self.linear_2_weight)
        torch.nn.init.xavier_uniform_(self.linear_3_weight)
        torch.nn.init.xavier_uniform_(self.linear_4_weight)

    def params(self):
        return (self.linear_1_weight, self.linear_1_bias, self.linear_2_weight, self.linear_2_bias,
                self.linear_3_weight, self.linear_3_bias, self.linear_4_weight, self.linear_4_bias)

    def unnormalized_log_probs_w_given_z_double_batched(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched_params(z, w,
                self.linear_1_weight,
                self.linear_1_bias,
                self.linear_2_weight,
                self.linear_2_bias,
                self.linear_3_weight,
                self.linear_3_bias,
                self.linear_4_weight,
                self.linear_4_bias)

    @staticmethod
    def energy_function_bilinear(i1, i2, W1, b1, W2, b2, W3, b3, W4, b4):
        batch_size = i1.shape[0]
        # outer_product = torch.einsum('bi,bj->bij', (i1, i2))
        # outer_product = outer_product.view(batch_size, -1)
        outer_product = torch.bmm(i1.unsqueeze(2), i2.unsqueeze(1)).view(batch_size, -1)
        dropout = torch.nn.Dropout(p=0.1)

        temp = torch.nn.functional.linear(outer_product, W1, bias=b1)
        temp = torch.nn.functional.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W2, b2)
        temp = torch.nn.functional.layer_norm(temp, temp.size()[1:])
        temp = torch.nn.functional.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W3, b3)
        temp = torch.nn.functional.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W4)

        return temp

    @staticmethod
    def energy(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedModelEmbeddingDynamics.energy_function_bilinear(z, w, W1, b1, W2, b2, W3, b3, W4, b4)

    @staticmethod
    def unnormalized_log_probs_w_given_z_params(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        return -EnergyBasedModelEmbeddingDynamics.energy(z, w, W1, b1, W2, b2, W3, b3, W4, b4)

    def unnormalized_log_probs_w_given_z(self, z, w):
        return -EnergyBasedModelEmbeddingDynamics.energy(z, w, self.linear_1_weight, self.linear_1_bias,
                                                         self.linear_2_weight, self.linear_2_bias,
                                                         self.linear_3_weight, self.linear_3_bias,
                                                         self.linear_4_weight, self.linear_4_bias)

    @staticmethod
    def unnormalized_log_probs_w_given_z_double_batched_params(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        first_dim = z.shape[0]
        second_dim = z.shape[1]
        dim = z.shape[2]
        energy = EnergyBasedModelEmbeddingDynamics.energy(z.reshape(-1, dim), w.reshape(-1, dim), W1, b1, W2, b2, W3, b3, W4, b4).reshape(first_dim, second_dim, 1)
        return -energy

    @staticmethod
    def expected_unnormalized_log_probs_w_given_z(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        samples_dim = z.shape[0]
        batch_dim = z.shape[1]
        dim = z.shape[2]
        energy = EnergyBasedModelEmbeddingDynamics.energy(z.reshape(-1, dim), w.reshape(-1, dim), W1, b1, W2, b2, W3, b3, W4, b4).reshape(samples_dim, batch_dim, 1)
        return -energy.mean(dim=0)

    # for small state spaces it is possible to manually compute the partition function
    @staticmethod
    def log_partition_function(z, W1, b1, W2, b2, W3, b3, W4, b4):
        dim = z.shape[-1]
        z = z.view(-1, 1, dim).expand(-1, 2**dim, -1)
        W = torch.arange(0, 2**dim, device=z.device).unsqueeze(-1).bitwise_and(2**torch.arange(dim, device=z.device)).ne(0).unsqueeze(0).expand(z.shape[0], -1, -1).float()
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched_params(z, W, W1, b1, W2, b2, W3, b3, W4, b4)
        partitions = torch.logsumexp(log_probs, dim=1)
        return partitions

    # the initial state will help create the optimal proposal distribution
    @staticmethod
    @torch.no_grad()
    def estimated_log_partition_function_better(z, initial_state, W, b, W1, b1, W2, b2, W3, b3, W4, b4, samples=16):
        z = z.expand(samples, -1, -1)
        initial_state = initial_state.expand(samples, -1, -1)
        w_batched = BoltzmannBasedEncoder.batched_encoder_sample(initial_state, W, b)

        proposal_log_probs = BoltzmannBasedEncoder.unnormalized_log_probs_a_given_b_double_batched_params(w_batched,
                                                                                                    initial_state,
                                                                                                    W, b, None, None)
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched_params(z, w_batched, W1,
                                                                                                    b1, W2, b2, W3, b3, W4, b4)
        return -math.log(samples) + torch.logsumexp(log_probs - proposal_log_probs, dim=0)

    # do some importance sampling here, note that this is probably good enough for training where the gradient can be noisy
    # and is in generally the right direction, but is definitely not good enough for evaluation
    @staticmethod
    @torch.no_grad()
    def estimated_log_partition_function(z, W1, b1, W2, b2, W3, b3, W4, b4, samples=512):
        dim = z.shape[-1]
        z = z.expand(samples, -1, -1)
        w = (torch.rand_like(z) < 0.5).float()
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched_params(z, w, W1, b1, W2,
                                                                                                    b2, W3, b3, W4, b4)
        return dim * math.log(2) - math.log(samples) + torch.logsumexp(log_probs, dim=0)

    def estimated_log_partition_function(self, z):
        return EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function(z, self.linear_1_weight,
                                                                                  self.linear_1_bias,
                                                                                  self.linear_2_weight,
                                                                                  self.linear_2_bias,
                                                                                  self.linear_3_weight,
                                                                                  self.linear_3_bias,
                                                                                  self.linear_4_weight,
                                                                                  self.linear_4_bias)

    @staticmethod
    def normalized_log_probabilities_w_given_z_params(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_params(z, w, W1, b1, W2,
                                                                                  b2, W3, b3, W4, b4) - \
               EnergyBasedModelEmbeddingDynamics.log_partition_function(z, W1, b1, W2, b2, W3, b3, W4, b4)

    def normalized_log_probabilities_w_given_z(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z_params(z, w, self.linear_1_weight,
                                                                                        self.linear_1_bias,
                                                                                        self.linear_2_weight,
                                                                                        self.linear_2_bias,
                                                                                        self.linear_3_weight,
                                                                                        self.linear_3_bias,
                                                                                        self.linear_4_weight,
                                                                                        self.linear_4_bias)

    def estimated_normalized_log_probabilities_w_given_z(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.estimated_normalized_log_probabilities_w_given_z_params(z, w, self.linear_1_weight,
                                                                                                  self.linear_1_bias,
                                                                                                  self.linear_2_weight,
                                                                                                  self.linear_2_bias,
                                                                                                  self.linear_3_weight,
                                                                                                  self.linear_3_bias,
                                                                                                  self.linear_4_weight,
                                                                                                  self.linear_4_bias)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z_params(z, w, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_params(z, w, W1, b1, W2,
                                                                                  b2, W3, b3, W4, b4) - \
               EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function(z, W1, b1, W2, b2, W3, b3, W4, b4)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z_better_params(z, w, x, W, b, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_params(z, w, W1, b1, W2,
                                                                                  b2, W3, b3, W4, b4) - \
               EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function_better(z, x, W, b, W1, b1, W2, b2, W3, b3, W4, b4)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z_better(z, w, x, model, samples=512):
        z_tilde = z.expand(samples, -1, -1)
        initial_state = x.expand(samples, -1, -1)

        # all from the proposal distribution
        w_tilde = model.encoder.batched_encoder_sample(initial_state)
        proposal_log_probs = model.encoder.unnormalized_log_probs_a_given_b_double_batched(w_tilde, initial_state)

        log_probs = model.embedding_dynamics.unnormalized_log_probs_w_given_z_double_batched(z_tilde, w_tilde)
        return model.embedding_dynamics.unnormalized_log_probs_w_given_z(z, w) + math.log(samples) - torch.logsumexp(
            log_probs - proposal_log_probs, dim=0)

class BoltzmannBasedEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        self.in_dim = in_dim # dimension of x
        self.out_dim = out_dim # dimension of w

        self.b = nn.Parameter(torch.zeros((1, self.out_dim)))
        self.W = nn.Parameter(torch.zeros((1, self.out_dim, self.in_dim)))
        self.padding1 = nn.Parameter(torch.zeros(1))
        self.padding2 = nn.Parameter(torch.zeros(1))

        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.b)
        torch.nn.init.xavier_uniform_(self.W)

    def params(self):
        return (self.b, self.W, self.padding1, self.padding2)

    @staticmethod
    def energy_function_linear(a, b, b_param, W_param):
        batch_size = b.shape[0]
        o = torch.nn.functional.logsigmoid((2 * a.view(batch_size, -1, 1) - 1) *
                                           (b_param.unsqueeze(-1) + W_param @ b.view(batch_size, -1,
                                                                                      1))).sum(dim=-2)
        return -o

    @staticmethod
    def energy(a, b, b_param, W_param):
        return BoltzmannBasedEncoder.energy_function_linear(a, b, b_param, W_param)

    @staticmethod
    def unnormalized_log_probs_a_given_b_params(a, b, b_param, W_param):
        return -BoltzmannBasedEncoder.energy(a, b, b_param, W_param)

    def unnormalized_log_probs_a_given_b(self, a, b):
        return BoltzmannBasedEncoder.unnormalized_log_probs_a_given_b_params(a, b, self.b,
                                                                      self.W)

    @staticmethod
    def encoder_sample(b, b_param, W_param):
        batch_size = b.shape[0]
        thresholds = torch.sigmoid(
            (b_param.unsqueeze(-1) + W_param @ b.view(batch_size, -1, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)

    def encoder_sample(self, b):
        batch_size = b.shape[0]
        thresholds = torch.sigmoid(
            (self.b.unsqueeze(-1) + self.W @ b.view(batch_size, -1, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)

    # simple since factorial distribution
    def batched_encoder_sample(self, x):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        return BoltzmannBasedEncoder.encoder_sample(x.reshape(dim_0 * dim_1, -1), self.W, self.b).reshape(dim_0,
                                                                                                         dim_1, -1)

    # simple since factorial distribution
    @staticmethod
    def batched_encoder_sample(x, W, b):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        return BoltzmannBasedEncoder.encoder_sample(x.reshape(dim_0 * dim_1, -1), W, b).reshape(dim_0, dim_1, -1)

    @staticmethod
    def unnormalized_log_probs_a_given_b_double_batched_params(a, b, b_param, W_param):
        first_dim = a.shape[0]
        second_dim = a.shape[1]
        dim_a = a.shape[2]
        dim_b = b.shape[2]
        energy = BoltzmannBasedEncoder.energy(a.reshape(-1, dim_a), b.reshape(-1, dim_b), b_param, W_param).reshape(
            first_dim, second_dim, 1)
        return -energy

    def unnormalized_log_probs_a_given_b_double_batched(self, a, b):
        first_dim = a.shape[0]
        second_dim = a.shape[1]
        dim_a = a.shape[2]
        dim_b = b.shape[2]
        energy = BoltzmannBasedEncoder.energy(a.reshape(-1, dim_a), b.reshape(-1, dim_b),
                                              self.b, self.W).reshape(
            first_dim, second_dim, 1)
        return -energy

    # for small state spaces it is possible to manually compute the partition function
    @staticmethod
    def log_partition_function(a_dim, b, b_param, W_param):
        print('this should not be called since its auto normalized')
        b_dim = b.shape[-1]
        b = b.view(-1, 1, b_dim).expand(-1, 2 ** a_dim, -1)
        A = torch.arange(0, 2 ** a_dim, device=b.device).unsqueeze(-1).bitwise_and(2 ** torch.arange(
            a_dim, device=b.device)).ne(0).unsqueeze(0).expand(b.shape[0], -1, -1).float()
        log_probs = BoltzmannBasedEncoder.unnormalized_log_probs_a_given_b_double_batched_params(A, b, b_param, W_param)
        partitions = torch.logsumexp(log_probs, dim=1)
        return partitions

    @staticmethod
    def conditional_log_probability_a_given_b_params(a, b, b_param, W_param):
        return BoltzmannBasedEncoder.unnormalized_log_probs_a_given_b_params(a, b, b_param, W_param)

    def conditional_log_probability_a_given_b(self, a, b):
        return BoltzmannBasedEncoder.conditional_log_probability_a_given_b_params(a, b, self.b,
                                                                           self.W)

    def conditional_log_probability_a_given_b_double_batched(self, a, b):
        dim_0 = a.shape[0]
        dim_1 = a.shape[1]
        return BoltzmannBasedEncoder.conditional_log_probability_a_given_b(a.reshape(dim_0 * dim_1, -1),
                                                                           b.reshape(dim_0 * dim_1, -1)).reshape(
            dim_0, dim_1, -1)


class EnergyBasedDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_ones=4):
        super().__init__()
        self.in_dim = in_dim # dimension of x
        self.out_dim = out_dim # dimension of w
        self.num_ones = num_ones
        if hidden_dim is None:
            hidden_dim = 32

        self.linear_1_weight = nn.Parameter(torch.zeros((hidden_dim, self.in_dim * self.out_dim)))
        self.linear_1_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_2_weight = nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.linear_2_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_3_weight = nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.linear_3_bias = nn.Parameter(torch.zeros((hidden_dim)))

        self.linear_4_weight = nn.Parameter(torch.zeros((1, hidden_dim)))
        self.linear_4_bias = nn.Parameter(torch.zeros((1)))
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.linear_1_weight)
        torch.nn.init.xavier_uniform_(self.linear_2_weight)
        torch.nn.init.xavier_uniform_(self.linear_3_weight)
        torch.nn.init.xavier_uniform_(self.linear_4_weight)

    def params(self):
        return (self.linear_1_weight, self.linear_1_bias, self.linear_2_weight, self.linear_2_bias,
                self.linear_3_weight, self.linear_3_bias, self.linear_4_weight, self.linear_4_bias)

    @staticmethod
    def energy_function_linear(a, b, W1, b1, W2, b2, W3, b3, W4, b4):
        outer_product = torch.einsum('bi,bj->bij', (a, b))
        dropout = torch.nn.Dropout(p=0.1)

        temp = torch.nn.functional.linear(outer_product.view(a.shape[0], -1), W1, b1)
        temp = torch.nn.functional.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W2, b2)
        temp = torch.nn.functional.layer_norm(temp, temp.size()[1:])
        temp = torch.nn.functional.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W3, b3)
        temp = torch.relu(temp)
        temp = dropout(temp)
        temp = torch.nn.functional.linear(temp, W4)
        return temp

    @staticmethod
    def energy(a, b, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedDecoder.energy_function_linear(a, b, W1, b1, W2, b2, W3, b3, W4, b4)

    @staticmethod
    def unnormalized_log_probs_a_given_b_params(num_ones, a, b, W1, b1, W2, b2, W3, b3, W4, b4):
        energy = EnergyBasedDecoder.energy(a, b, W1, b1, W2, b2, W3, b3, W4, b4)
        return torch.where(a.sum(dim=-1, keepdim=True) == num_ones, -energy, -torch.ones_like(energy) * float('inf'))

    def unnormalized_log_probs_a_given_b(self, a, b):
        return EnergyBasedDecoder.unnormalized_log_probs_a_given_b_params(self.num_ones, a, b, self.linear_1_weight, self.linear_1_bias,
                                                                        self.linear_2_weight, self.linear_2_bias,
                                                                        self.linear_3_weight, self.linear_3_bias,
                                                                        self.linear_4_weight, self.linear_4_bias)

    # MAKE SURE WE DON’T USE THIS ANYMORE, USELESS UNLESS WE ENFORCE OUT OF DOMAIN SAMPLES TO BE -INF
    @staticmethod
    def unnormalized_log_probs_a_given_b_double_batched_params(num_ones, a, b, W1, b1, W2, b2, W3, b3, W4, b4):
        first_dim = a.shape[0]
        second_dim = a.shape[1]
        dim_a = a.shape[2]
        dim_b = b.shape[2]
        energy = EnergyBasedDecoder.energy(a.reshape(-1, dim_a), b.reshape(-1, dim_b), W1, b1, W2, b2, W3, b3, W4, b4).reshape(
            first_dim, second_dim, 1)
        return torch.where(a.sum(dim=-1, keepdim=True) == num_ones, -energy, -torch.ones_like(energy) * float('inf'))

    def unnormalized_log_probs_a_given_b_double_batched(self, a, b):
        first_dim = a.shape[0]
        second_dim = a.shape[1]
        dim_a = a.shape[2]
        dim_b = b.shape[2]
        energy = EnergyBasedDecoder.energy(a.reshape(-1, dim_a), b.reshape(-1, dim_b),
                                           self.linear_1_weight, self.linear_1_bias,
                                           self.linear_2_weight, self.linear_2_bias,
                                           self.linear_3_weight, self.linear_3_bias,
                                           self.linear_4_weight, self.linear_4_bias).reshape(
            first_dim, second_dim, 1)
        return torch.where(a.sum(dim=-1, keepdim=True) == self.num_ones, -energy, -torch.ones_like(energy) * float('inf'))

    @staticmethod
    def expected_unnormalized_log_probs_a_given_b(num_ones, a, b, W1, b1, W2, b2, W3, b3, W4, b4):
        log_probs = EnergyBasedDecoder.unnormalized_log_probs_a_given_b_double_batched_params(num_ones, a, b, W1, b1, W2, b2, W3, b3, W4, b4)
        return log_probs.mean(dim=0)

    @staticmethod
    def log_partition_function(num_ones, a_dim, b, W1, b1, W2, b2, W3, b3, W4, b4):
        b_dim = b.shape[-1]
        b = b.view(-1, 1, b_dim).expand(-1, 2 ** a_dim, -1)
        bit_strings = torch.arange(0, 2 ** a_dim, device=b.device).unsqueeze(-1).bitwise_and(2 ** torch.arange(
            a_dim, device=b.device)).ne(0)

        A = bit_strings.unsqueeze(0).expand(b.shape[0], -1, -1).float()
        log_probs = EnergyBasedDecoder.unnormalized_log_probs_a_given_b_double_batched_params(num_ones, A, b, W1, b1, W2, b2, W3, b3, W4, b4)
        partitions = torch.logsumexp(log_probs[:, bit_strings.sum(dim=-1) == num_ones, :], dim=1)
        return partitions

    @staticmethod
    @torch.no_grad()
    def estimate_log_partition_function(num_ones, b, initial_state, W1, b1, W2, b2, W3, b3, W4, b4, samples=512):
        # samples, batch_size, dim
        batch_size = b.shape[0]
        b = b.expand(samples, -1, -1)
        initial_state = initial_state.expand(samples, -1, -1)
        importance_samples = torch.zeros_like(initial_state)

        # these will become the importance samples
        x = np.arange(initial_state.shape[-1])
        perms = rng.permuted(np.tile(x, [samples, batch_size]).reshape(samples, batch_size, x.size), axis=-1)[..., :num_ones]
        importance_samples = torch.zeros_like(initial_state)
        importance_samples[np.arange(samples).reshape(samples, 1, 1), np.arange(batch_size).reshape(batch_size, 1), perms] = 1.0
        
        # importance sampling probabilities are fixed by the input distribution, which is uniform
        dim = initial_state.shape[-1]
        proposal_log_probs = -math.log(math.comb(dim, num_ones))
        
        log_probs = EnergyBasedDecoder.unnormalized_log_probs_a_given_b_double_batched_params(num_ones, importance_samples, b, W1,
                                                                                       b1, W2, b2, W3, b3, W4, b4)
        return -math.log(samples) + torch.logsumexp(log_probs - proposal_log_probs, dim=0)


    @staticmethod
    def conditional_log_probability_a_given_b_params(a, b, num_ones, W1, b1, W2, b2, W3, b3, W4, b4):
        log_probs = EnergyBasedDecoder.unnormalized_log_probs_a_given_b_params(num_ones, a, b, W1, b1, W2,
                                                                        b2, W3, b3, W4, b4) - EnergyBasedDecoder.log_partition_function(num_ones, a.shape[-1],
                                                                                                                                       b, W1, b1, W2, b2, W3, b3, W4, b4)
        return log_probs


    @staticmethod
    def estimated_conditional_log_probability_a_given_b(a, b, num_ones, W1, b1, W2, b2, W3, b3, W4, b4):
        return EnergyBasedDecoder.unnormalized_log_probs_a_given_b_params(num_ones, a, b, W1, b1, W2,
                                                                   b2, W3, b3, W4, b4) - EnergyBasedDecoder.estimate_log_partition_function(num_ones, b, a, W1,
                                                                                                                                                  b1, W2, b2, W3, b3, W4, b4, samples=4096)


    def conditional_log_probability_a_given_b(self, a, b):
        return EnergyBasedDecoder.conditional_log_probability_a_given_b_params(a, b, self.num_ones, self.linear_1_weight, self.linear_1_bias,
                                                                         self.linear_2_weight, self.linear_2_bias,
                                                                         self.linear_3_weight, self.linear_3_bias,
                                                                         self.linear_4_weight, self.linear_4_bias)


    def conditional_log_probability_a_given_b_double_batched(self, a, b):
        dim0 = a.shape[0]
        dim1 = a.shape[1]
        return EnergyBasedDecoder.conditional_log_probability_a_given_b(a.reshape(dim0 * dim1, -1),
                                                                         b.reshape(dim0 * dim1, -1)).reshape(
            dim0, dim1, -1)



