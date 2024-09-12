import torch
import numpy as np
import skfda.representation.basis as basis
import torch.nn as nn

def tdnFunNN_test(data,model,number_of_basis_function,K,J,L,batch_size,epochs):
    number_of_tensor_1 = data.size()[0]
    number_of_tensor_2 = data.size()[1]
    number_of_tensor_3 = data.size()[2]
    # -----------------------------------------------------------------------------------------------------------------------
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

    y_test=data

    class Find_Z(nn.Module):
        def __init__(self, input_score_number, model, input_size, basis_eval, basis_eval1, basis_eval_number,
                     basis_eval1_number, submodel_number):
            super(Find_Z, self).__init__()
            # Basis eval input
            self.input_score_number = input_score_number

            self.input_layer = nn.Linear(input_size, input_score_number, bias=False)
            self.fc1 = model.fc1
            self.fc2 = model.fc2
            self.model = model
            self.basis_eval = basis_eval
            self.basis_eval1 = basis_eval1.t()
            self.basis_eval_number = basis_eval_number
            self.basis_eval1_number = basis_eval1_number

            self.ReLU = nn.ReLU()
            self.Sigmoid = nn.Sigmoid()
            self.weigths_linear_combine = model.weigths_linear_combine
            self.bias_linear = model.bias_linear
            self.submodel_number = submodel_number

        def forword(self, x, input_number):
            m = self.basis_eval_number
            n = self.basis_eval1_number
            k = self.submodel_number

            out0 = self.input_layer(x)
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
    input_score_number = K
    input_matrix = torch.eye(len(y_test))
    model1 = Find_Z(input_score_number=input_score_number, model=model, input_size=len(y_test), basis_eval=basis_eval,
                    basis_eval1=basis_eval1, basis_eval_number=basis_eval_number, basis_eval1_number=basis_eval1_number,
                    submodel_number=8)
    for name, p in model1.named_parameters():
        if name.startswith('fc1'): p.requires_grad = False
        if name.startswith('fc2'): p.requires_grad = False
    optimizer1 = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, model1.parameters()), lr=0.001)

    for i in range(epochs):
        batch_loss = []
        # Mini-Batch方法进行训练
        for start in range(0, len(y_test), batch_size):
            end = start + batch_size if start + batch_size < len(y_test) else len(y_test)
            yy = torch.tensor(y_test[start:end, :, :], dtype=torch.float)
            prediction = model1.forword(input_matrix[start:end], len(range(start, end)))
            loss = sum(sum(sum((prediction - yy) ** 2))) / (len(s) * len(t))
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            batch_loss.append(loss.data.numpy())
        # 打印损失
        if i % 100 == 0:
            error_tem = sum(sum(sum((y_test - model1.forword(input_matrix, len(y_test))) ** 2)))
            error_tem = error_tem / (len(s) * len(t) * len(y_test))
            print(i, "Find", torch.sqrt(error_tem))

    data_simulation_test=data
    error1_rmse_2 = sum(sum(sum(((data_simulation_test - model1.forword(input_matrix, len(data_simulation_test))) ** 2))))
    error1_rmse_2 = error1_rmse_2 / (len(s) * len(t) * len(data_simulation_test))
    rmse_test = torch.sqrt(error1_rmse_2)

    error1_rrmse_2 = sum(
        sum(sum((data_simulation_test - model1.forword(input_matrix, len(data_simulation_test))) ** 2)))
    rrmse_test= torch.sqrt(error1_rrmse_2) / torch.sqrt(sum(sum(sum((data_simulation_test) ** 2))))

    return rmse_test,rrmse_test
