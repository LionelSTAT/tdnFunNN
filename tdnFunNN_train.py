import torch
import numpy as np
import skfda.representation.basis as basis
import torch.nn as nn

def tdnFunNN_train(data,number_of_basis_function,K,J,L,batch_size,epochs):

    number_of_tensor_1 = data.size()[0]
    number_of_tensor_2 = data.size()[1]
    number_of_tensor_3 = data.size()[2]
#-----------------------------------------------------------------------------------------------------------------------
    s = torch.linspace(0, 1, number_of_tensor_2)
    t = torch.linspace(0, 1, number_of_tensor_3)

    basis_eval_number = number_of_basis_function
    basis_eval1_number = number_of_basis_function
    basis_eval_grid = np.linspace(0, 1, num=len(s))
    basis_eval_grid1 = np.linspace(0, 1, num=len(t))
    basis_eval = basis.BSplineBasis(n_basis=basis_eval_number).evaluate(basis_eval_grid)[:, :, 0].T
    basis_eval1 = basis.BSplineBasis(n_basis=basis_eval1_number).evaluate(basis_eval_grid1)[:, :, 0].T
    basis_eval = torch.tensor(basis_eval)
    basis_eval = torch.tensor(basis_eval, dtype=torch.float32)
    basis_eval1 = torch.tensor(basis_eval1)
    basis_eval1 = torch.tensor(basis_eval1, dtype=torch.float32)
#-----------------------------------------------------------------------------------------------------------------------
    class HDNLPCA(nn.Module):
        def __init__(self, input_size, input_score_number, basis_eval, basis_eval_number, basis_eval1,
                     basis_eval1_number,
                     submodel_number, linear_combination_layer_number):
            super(HDNLPCA, self).__init__()
            # Basis eval input
            self.basis_eval = basis_eval
            self.basis_eval1 = basis_eval1.t()

            # Activation function
            self.Sigmoid = nn.Sigmoid()
            self.ReLU = nn.ReLU()

            # input matrix to input parameters
            self.fc0 = nn.Linear(input_size, input_score_number, bias=False)

            # First fully connected layer
            self.fc1 = nn.Linear(input_score_number, basis_eval_number * basis_eval1_number * submodel_number)
            self.fc2 = nn.Linear(basis_eval_number * basis_eval1_number * submodel_number,
                                 basis_eval_number * basis_eval1_number * submodel_number)

            # Linear combination Layer
            # self.linear_matrix = nn.Parameter(
            #     torch.randn((submodel_number, submodel_number * linear_combination_layer_number)))
            # self.weights = nn.Parameter(torch.randn((submodel_number, 1)))
            # self.bias = nn.Parameter(torch.randn(basis_eval_number, basis_eval1_number))

            self.weigths_linear_combine = nn.Parameter(torch.randn(1, submodel_number))
            self.weights_linear_combine1 = nn.Parameter(torch.randn(1, submodel_number))
            self.bias_linear = nn.Parameter(torch.randn(basis_eval_number, basis_eval1_number))

            # Parameters
            self.basis_eval_number = basis_eval_number
            self.basis_eval1_number = basis_eval1_number
            self.submodel_number = submodel_number
            self.linear_combination_layer_number = linear_combination_layer_number

        def forward(self, x, input_number):
            m = self.basis_eval_number
            n = self.basis_eval1_number
            k = self.submodel_number

            out0 = self.fc0(x)
            out1 = self.fc1(out0)
            out1 = self.fc2(out1)

            out2 = out1.reshape(input_number, n * k, m)
            out2 = torch.transpose(out2, dim0=1, dim1=2)

            out3 = torch.matmul(self.basis_eval, out2)

            basis_eval1_expand = torch.zeros(n * k, k * self.basis_eval1.size()[1])
            for i in range(k):
                basis_eval1_expand[n * i:n * (i + 1),
                (self.basis_eval1.size()[1] * i):(self.basis_eval1.size()[1] * (i + 1))] = self.basis_eval1

            out3 = torch.matmul(out3, basis_eval1_expand)

            out3_activation = self.Sigmoid(out3)

            out_final = torch.zeros(out3_activation.size()[0], out3_activation.size()[1],
                                    int(out3_activation.size()[2] / k))

            for i in range(k):
                out_final = out_final + out3_activation[:, :,
                                        self.basis_eval1.size()[1] * i:self.basis_eval1.size()[1] * (i + 1)] * \
                            self.weigths_linear_combine[0, i]
            # out_final = out_final + torch.matmul(torch.matmul(self.basis_eval, self.bias_linear), self.basis_eval1)
            return out_final


    losses = []
    batch_size = batch_size
    y = data
    input_score_number = K
    input_matrix = torch.eye(len(y))

    model = HDNLPCA(input_size=len(y), input_score_number=K, basis_eval=basis_eval,
                    basis_eval_number=number_of_basis_function,
                    basis_eval1=basis_eval1, basis_eval1_number=number_of_basis_function,
                    submodel_number=J, linear_combination_layer_number=L)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    for i in range(epochs):
        batch_loss = []
        # Mini-Batch方法进行训练
        for start in range(0, len(y), batch_size):
            end = start + batch_size if start + batch_size < len(y) else len(y)
            yy = torch.tensor(y[start:end, :, :], dtype=torch.float)
            prediction = model(input_matrix[start:end], len(range(start, end)))
            loss = sum(sum(sum((prediction - yy) ** 2))) / (len(s) * len(t))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())
        if i % 100 == 0:
            error_tem = sum(sum(sum((y - model(input_matrix, len(y))) ** 2)))
            error_tem = error_tem / (len(s) * len(t) * len(y))
            print(i, torch.sqrt(error_tem.mean()))

    score = model.fc0(input_matrix)
    data_simulation=data
    error1_rmse_1 = sum(sum(sum((data_simulation - model(input_matrix, len(data_simulation))) ** 2)))
    error1_rmse_1 = error1_rmse_1 / (len(s) * len(t) * len(data_simulation))
    rmse = torch.sqrt(error1_rmse_1)

    error1_rrmse_1 = sum(sum(sum((data_simulation - model(input_matrix, len(data_simulation))) ** 2)))
    rrmse = torch.sqrt(error1_rrmse_1) / torch.sqrt(sum(sum(sum((data_simulation) ** 2))))

    return rmse,rrmse,score,model

tdnFunNN_train(data_simulation,10,2,8,4,32,1000)


###Another data select
s = torch.linspace(0, 1, 20)
t = torch.linspace(0, 1, 20)
simulation_sample_number=1000
basis_eval_number = 10
basis_eval1_number = 10
basis_eval_grid = np.linspace(0, 1, num=len(s))
basis_eval_grid1 = np.linspace(0, 1, num=len(t))
basis_eval = basis.BSplineBasis(n_basis=basis_eval_number).evaluate(basis_eval_grid)[:, :, 0].T
basis_eval1 = basis.BSplineBasis(n_basis=basis_eval1_number).evaluate(basis_eval_grid1)[:, :, 0].T

###Generate simulation Data######
Omage_1 = torch.ones(int(simulation_sample_number / 2)) * 2
#Omage_1 = torch.ones(int(simulation_sample_number / 2)) * 4

Z_1a = torch.randn(int(simulation_sample_number / 2)) * 0.2 + 1
theta_1a = torch.rand(int(simulation_sample_number / 2)) * 2 * torch.pi
Z_2a = torch.randn(int(simulation_sample_number / 2)) * 0.2 + 4
theta_2a = torch.rand(int(simulation_sample_number / 2)) * 2 * torch.pi
data_simulation1 = torch.zeros(int(simulation_sample_number / 2), 20, 20)
for i in range(int(simulation_sample_number / 2)):
    x_lab = Z_1a[i] * torch.sin(theta_1a[i]) * torch.cos(Omage_1[i] * torch.pi * s)
    y_lab = Z_1a[i] * torch.cos(theta_1a[i]) * torch.sin(Omage_1[i] * torch.pi * t)
    for j in range(20):
        data_simulation1[i, j, :] = data_simulation1[i, j, :] + x_lab
        data_simulation1[i, :, j] = data_simulation1[i, :, j] + y_lab

data_crossanderror1 = torch.zeros(int(simulation_sample_number / 2), 20, 20)
for i in range(20):
    for j in range(20):
        data_crossanderror1[:, i, j] = torch.ones(int(simulation_sample_number / 2)) * Z_2a *torch.sin(theta_1a[i])* \
                                               torch.cos(Omage_1*s[i]) * torch.sin(Omage_1*t[j])

data_noerror1=data_simulation1+data_crossanderror1
data_crossanderror1 = data_crossanderror1 + torch.randn(int(simulation_sample_number / 2), 20, 20) * 0.1
data_simulation1 = data_simulation1 + data_crossanderror1

data_simulation=data_simulation1
data_simulation = data_simulation - torch.mean(data_simulation, dim=0)

data_simulation_noerror=data_noerror1
data_simulation_noerror=data_simulation_noerror-torch.mean(data_simulation_noerror, dim=0)

Z_1b = torch.randn(int(simulation_sample_number / 2)) * 0.2 + 1
theta_1b = torch.rand(int(simulation_sample_number / 2)) * 2 * torch.pi
Z_2b = torch.randn(int(simulation_sample_number / 2)) * 0.2 + 4
theta_2b = torch.rand(int(simulation_sample_number / 2)) * 2 * torch.pi


data_simulation3 = torch.zeros(int(simulation_sample_number / 2), 20, 20)
    for i in range(int(simulation_sample_number / 2)):
        x_lab = Z_1b[i] * torch.sin(theta_1b[i]) * torch.cos(Omage_1[i] * torch.pi * s)
        y_lab = Z_1b[i] * torch.cos(theta_1b[i]) * torch.sin(Omage_1[i] * torch.pi * t)
        for j in range(20):
            data_simulation3[i, j, :] = data_simulation3[i, j, :] + x_lab
            data_simulation3[i, :, j] = data_simulation3[i, :, j] + y_lab

    data_crossanderror3 = torch.zeros(int(simulation_sample_number / 2), 20, 20)
    for i in range(20):
        for j in range(20):
            data_crossanderror3[:, i, j] = torch.ones(int(simulation_sample_number / 2)) * Z_2b *torch.sin(theta_1a[i])* \
                                               torch.cos(Omage_1*s[i]) * torch.sin(Omage_1*t[j])

    data_noerror3=data_simulation3+data_crossanderror3
    data_crossanderror3 = data_crossanderror3 + torch.randn(int(simulation_sample_number / 2), 20, 20) * 0.1
    data_simulation3 = data_simulation3 + data_crossanderror3

    data_simulation_noerror_test=data_noerror3
    data_simulation_noerror_test=data_simulation_noerror_test-torch.mean(data_simulation_noerror_test,dim=0)


    data_simulation_test = data_simulation3
    data_simulation_test = data_simulation_test - torch.mean(data_simulation_test, dim=0)