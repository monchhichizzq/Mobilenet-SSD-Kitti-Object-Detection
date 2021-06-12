# -*- coding: utf-8 -*-
# @Author: zzhu
# @Date:   2021-06-02 16:12:31
# @Last Modified by:   zzhu
# @Last Modified time: 2021-06-02 17:40:11

import os
import numpy as np
import matplotlib.pyplot as plt

class Train_Visualizer():
	def __init__(self, **kwargs):
		self.model_folder = kwargs.get('model_folder', None)
		save = kwargs.get('save_npy', 'save_data.npy')
		self.save_folder = kwargs.get('save_folder', 'save_folder')
		os.makedirs(self.save_folder, exist_ok=True)
		self.save = os.path.join(self.save_folder, save)

		self.epochs = []
		self.mAPs = []
		self.nb_spikes = []
		self.tra_losses = []
		self.val_losses = []
		self.box_losses = []
		self.rl1_losses = []

	def read_data(self):
		for file in os.listdir(self.model_folder):
			'''
				_, train_loss, val_loss, bbx_loss, l1_loss, mAP, val_nb_spikes
			'''

			components = file.split('-')
			epoch = int(components[0].split('_')[-1])
			tra_loss = float(components[1].split('_')[-1])
			val_loss = float(components[2].split('_')[-1])
			bbx_loss = float(components[3].split('_')[-1])
			rl1_loss = float(components[4].split('_')[-1])
			mAP_loss = float(components[5].split('_')[-1])
			nb_spike = float(components[6].split('_')[-1][:-3])

			self.epochs.append(epoch)
			self.mAPs.append([epoch, mAP_loss])
			self.nb_spikes.append([epoch, nb_spike])
			self.tra_losses.append([epoch, tra_loss])
			self.val_losses.append([epoch, val_loss])
			self.box_losses.append([epoch, bbx_loss])
			self.rl1_losses.append([epoch, rl1_loss])

	def save_data(self):
		save_data_array = np.array([self.mAPs, 
									self.nb_spikes,
									self.tra_losses,
									self.val_losses,
									self.box_losses, 
									self.rl1_losses])
		np.save(self.save, save_data_array)
		print(save_data_array)

	def sort_data(self, data_list):
		data_list = sorted(data_list, key=lambda x: x[0])
		sort_data_list = [m[1] for m in data_list]
		epochs_list = [m[0] for m in data_list]
		return epochs_list, sort_data_list

	def plot_mAPs(self):
		epochs, mAPs = self.sort_data(self.mAPs)
		save_fig = os.path.join(self.save_folder, 'train_mAPs.png')
		plt.plot(epochs, mAPs)
		plt.xlabel('Epoch')
		plt.ylabel('mAP %')
		plt.savefig(save_fig, dpi=600)
		plt.show()

	def plot_spikes(self):
		epochs, nb_spikes = self.sort_data(self.nb_spikes)
		save_fig = os.path.join(self.save_folder, 'train_spikes.png')
		plt.plot(epochs, nb_spikes)
		plt.xlabel('Epoch')
		plt.ylabel('Spikes (M)')
		plt.savefig(save_fig, dpi=600)
		plt.show()

	def plot_losses(self):
		epochs, tra_losses = self.sort_data(self.tra_losses)
		epochs, val_losses = self.sort_data(self.val_losses)
		epochs, box_losses = self.sort_data(self.box_losses)
		epochs, rl1_losses = self.sort_data(self.rl1_losses)
		save_fig = os.path.join(self.save_folder, 'train_losses.png')
		plt.plot(epochs, tra_losses, label='train')
		plt.plot(epochs, val_losses, label='val')
		plt.plot(epochs, box_losses, label='bbox')
		plt.plot(epochs, rl1_losses, label='l1')
		plt.legend()
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.savefig(save_fig, dpi=600)
		plt.show()
  
	def __call__(self, target_metric):

		self.target_metric = target_metric

		self.read_data()
		if self.save:
			self.save_data()

		if self.target_metric == 'mAP':
			self.plot_mAPs()
		if self.target_metric == 'nb_spikes':
			self.plot_spikes()
		if self.target_metric == 'losses':
			self.plot_losses()




if __name__ == '__main__':
    

    v = Train_Visualizer(save_folder = 'mbv1_ssd_l1-7_thresh-7',
    					 save_npy = 'save_data.npy',
    					 model_folder = '../Trainer_tfdata_mb/logs_v1/mbv1_thresh-7_l11e-07',
    					 )
    v(target_metric = 'losses')
    v(target_metric = 'nb_spikes')
    v(target_metric = 'mAP')
    


