import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from scipy import linalg

N = 100
T = 10.0
dt = T/N
nVars = 4
nControls = 1
batch = (nVars + nControls)*3


def getNewState():
 # we're going to also pack f into x
	x = torch.zeros(batch, N, nVars+nControls, requires_grad=True)
	l = torch.zeros(batch, N-1, nVars, requires_grad=False)
	return x, l


def calc_loss(x, l, rho, prox=0):  # l,
	# depack f, it has one less time point
	cost = 0.1*torch.sum(x**2)
	f = x[:, 1:, :nControls]
	x = x[:, :, nControls:]

	del_x = (x[:, 1:, :] - x[:, :-1, :]) / dt
	x_bar = (x[:, 1:, :] + x[:, :-1, :]) / 2

	dxdt = torch.zeros(batch, N-1, nVars)
	
	POS = 0
	VEL = 1
	THETA = 2
	OMEGA = 3
	
	dxdt[:, :, POS] = x_bar[:, :, VEL]
	#print(dxdt.shape)
	#print(f.shape)
	dxdt[:, :, VEL] = f[:, :, 0]
	dxdt[:, :, THETA] = x_bar[:, :, OMEGA]
	dxdt[:, :, OMEGA] = - \
		torch.sin(x_bar[:, :, THETA]) + f[:, :, 0]*torch.cos(x_bar[:, :, THETA])


	xdot_res = del_x - dxdt

	lagrange_mult = torch.sum(l * xdot_res)
	penalty = rho*torch.sum(xdot_res**2)

	cost += 1.0*torch.sum((x[:, :, THETA]-np.pi)**2 * torch.arange(N) / N)
	cost += 0.5*torch.sum(f**2)

	return cost, penalty, lagrange_mult, xdot_res


def getGradHessBand(loss, B, x):
	#B = bandn
	delL0, = torch.autograd.grad(loss, x, create_graph=True)
	delL = delL0[:, 1:, :].view(B, -1)  # remove x0
	print("del ", delL[:, :10])
	#hess = torch.zeros(B,N-1,nVars+nControls, requires_grad=False).view(B,B,-1)
	y = torch.zeros(B, N-1, nVars+nControls, requires_grad=False).view(B, -1)

	#y = torch.eye(B).view(B,1,B)
	#print(y.shape)
	for i in range(B):
	#y = torch.zeros(N-1,nVars+nControls, requires_grad=False).view(-1)
		y[i, i::B] = 1
		#print(y[:,:2*B])
	print(y.shape)
	print(delL.shape)
	delLy = torch.sum(delL * y)
	delLy.backward()  # (i != B-1)


	#print(hess)
	# .view(-1)# hess.detach().numpy()
	nphess = x.grad[:, 1:, :].view(B, -1).detach().numpy()
	#print(nphess[:,:4])
	#print(nphess)
	for i in range(B):
		nphess[:, i::B] = np.roll(nphess[:, i::B], -i+B//2, axis=0)
		print(nphess[:, :4])
	#hessband = removeX0(nphess[:B//2+1,:])
	#grad = removeX0(delL.detach().numpy())
	return delL.detach().numpy()[0, :], nphess  # hessband


def plot(xdot_res, x):
	plt.subplot(121)
	plt.plot(xdot_res[0, :, 0].detach().numpy(), label='POSdot_res')
	plt.plot(xdot_res[0, :, 1].detach().numpy(), label='VELdot_res')
	plt.plot(xdot_res[0, :, 2].detach().numpy(), label='THETAdot_res')
	plt.plot(xdot_res[0, :, 3].detach().numpy(), label='OMEGAdot_res')
	plt.legend(loc='upper right')

	plt.subplot(122)
	plt.plot(x[0, :, 1].detach().numpy(), label='POS')
	plt.plot(x[0, :, 2].detach().numpy(), label='VEL')
	plt.plot(x[0, :, 3].detach().numpy(), label='Theta')
	plt.plot(x[0, :, 4].detach().numpy(), label='OMEGA')
	plt.plot(x[0, :, 0].detach().numpy(), label='F')
	#plt.plot(cost[0,:].detach().numpy(), label='F')
	plt.legend(loc='upper right')
	#plt.figure()
	#plt.subplot(133)
	#plt.plot(costs)

	plt.show()


def main():
	x, l = getNewState()
	rho = 0.1
	prox = 0.0
	for j in range(10):
		while True:
			try:
				cost, penalty, lagrange_mult, xdot_res = calc_loss(x, l, rho, prox)
				#print(total_cost)
				print("hey now")
				#print(cost)
				total_cost = cost + lagrange_mult + penalty
				#total_cost = cost
				gradL, hess = getGradHessBand(total_cost, (nVars+nControls)*3, x)
				#print(hess)
				#print(hess.shape)
				gradL = gradL.reshape(-1)
				#print(gradL.shape)

				#easiest thing might be to put lagrange mutlipleirs into x.
				#Alternatively, use second order step in penalty method.
				bandn = (nVars+nControls)*3//2
				print(hess.shape)
				print(gradL.shape)
				dx = linalg.solve_banded((bandn, bandn), hess, gradL)
				x.grad.data.zero_()
				#print(hess)
				#print(hess[bandn:,:])
				#dx = linalg.solveh_banded(hess[:bandn+1,:], gradL, overwrite_ab=True)
				newton_dec = np.dot(dx, gradL)
				#df0 = dx[:nControls].reshape(-1,nControls)
				dx = dx.reshape(1, N-1, nVars+nControls)

				with torch.no_grad():
					x[:, 1:, :] -= torch.tensor(dx)
					print(x[:, :5, :])
				#print(x[:,0,nVars:].shape)
				#print(df0.shape)
				costval = cost.detach().numpy()
				#break
				if newton_dec < 1e-10*costval:
					break
			except np.linalg.LinAlgError:
				print("LINALGERROR")
		prox += 0.1
		#break
		#print(x)
		with torch.no_grad():
			l += 2 * rho * xdot_res
			rho = rho * 2  # + 0.1

	plot(xdot_res, x)

if __name__ == "__main__":
	main()
