from pathlib import Path
from torch.autograd import Variable
import torch
import os
import cv2
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from utils import clear, count_sliding_window, make_optimizer, make_scheduler, CrossEntropy2d, accuracy, metrics, sliding_window, grouper, convert_from_color, convert_to_color, global_accuracy
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
# from focal_loss.focal_loss import FocalLoss

class Trainer():
    
    def __init__(self, net, loader, params, scheduler = True, cbkp = None,):
        
        self.net = net
        self.loader = loader
        self.params = params
        self.iter_ = 0
        self.losses = np.zeros(1000000)
        self.mean_losses = np.zeros(100000000)
        self.print_each = 100 # Print statistics every 500 iterations
        
        # Weights for class balancing
        self.weight_cls = self.prepare([self.params['weights']])
        
        # Define an id to a trained model. Use the number of seconds since 1970
        time_ = str(time.time())
        time_ = time_.replace(".", "")
        self.model_id = time_

        # Create optimizer
        self.optimizer = make_optimizer(self.params['optimizer_params'], self.net)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_loss = 0.0
        self.last_epoch = 0.0

        # Load a previously model if it exists
        if cbkp is not None:
            self.load(cbkp)
            
    def load(self, path):
        
        # Check if model file exists
        assert os.path.exists(path), "{} cant be opened".format(path)
        
        checkpoint = torch.load(path)

        self.last_epoch = checkpoint['epoch']
        self.last_loss = checkpoint['loss']
        self.model_id = checkpoint['model_id']
        try:
            self.mean_losses = checkpoint['mean_losses']
        except KeyError:
            self.mean_losses = np.zeros(100000000)
            
        print(self.last_epoch)
        
        # Load model and optimizer params
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            for _ in range(0, self.last_epoch): self.scheduler.step()

    def save(self, path = None):

        if path is None:
            path = './{}_model_final.pth.tar'.format(self.model_id)

        # Save current loss, epoch, model weights and optimizer params
        torch.save({
            'epoch': self.last_epoch,
            'loss': self.last_loss,
            'mean_losses': self.mean_losses,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.params['cpu'] is not None else 'cuda') # Define run-device torch
        def _prepare(tensor):
            if self.params['precision'] == 'half': tensor = tensor.half() # Convert to half precision
            return tensor.to(device)
        return [_prepare(_l) for _l in l]

    def test(self, test_loader = None, stride = None, window_size = None, batch_size = None, all=False):
    
        # Loading test parameters
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"

        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"

        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"

        bs = batch_size if batch_size is not None else self.params['bs']
        bs = bs if bs is not None else 1

        # Get all data in test data loader
        input_ids, label_ids = test_ld.dataset.get_dataset()
        # test_inputs_name = np.copy(input_ids)

        # Use the network on the test set
        test_images = (1 / 255 * np.asarray(io.imread(id)[:,:,:3], dtype='float32') for id in input_ids)
        test_labels = (np.asarray(io.imread(id)[:,:,:3], dtype='uint8') for id in label_ids)
        eroded_labels = (convert_from_color(io.imread(id)[:,:,:3]) for id in label_ids)
        
        all_preds = []
        all_gts = []
        
        # Switch the network to inference mode
        self.net.eval()
        
        for img, gt, gt_e, image_path in tqdm(zip(test_images, test_labels, eroded_labels, input_ids), total=len(input_ids), leave=False):
            print(f'Image Path -> {image_path}')
            filepath = os.path.split(image_path)[1].split('.')[0]
            print(filepath)
            Path(f'./tmp/{filepath}').mkdir(parents=True, exist_ok=True)
            
            pred = np.zeros(img.shape[:2] + (self.params['n_classes'],))

            total = count_sliding_window(img, step=test_stride, window_size=test_ws) // bs
            for i, coords in enumerate(tqdm(grouper(bs, sliding_window(img, step=test_stride, window_size=test_ws)), total=total, leave=False)):
                # Display in progress results
                if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1) #Usar argmax do pytorch
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    clear()
                    # plt.show()
                    fig.savefig(f"./tmp/test_progress", dpi=fig.dpi, bbox_inches='tight')
                        
                # Build the tensor (ex: 2048px/256px = 8 patches de 256x256)
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                
                # Do the inference
                outs = self.net(image_patches)
                outs = outs.data.cpu().numpy()
                # trocar pred pela função do pytorch
                
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0)) # Usar transpose do Pytorch
                    pred[x:x+w, y:y+h] += out
                del(outs)

            pred = np.argmax(pred, axis=-1)

            # Display the result
            clear()
            # print(f'--- {input_ids} ---')
            
            # fig = plt.figure()
            # fig.add_subplot(1,3,1)
            # plt.imshow(np.asarray(255 * img, dtype='uint8'))
            # fig.add_subplot(1,3,2)
            # plt.imshow(convert_to_color(pred))
            # fig.add_subplot(1,3,3)
            # plt.imshow(gt)
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.set_title('RGB')
            plt.imshow(np.asarray(255 * img, dtype='uint8'))
            ax2 = fig.add_subplot(1,3,2)
            ax2.set_title('Pred')
            plt.imshow(convert_to_color(pred))
            ax3 = fig.add_subplot(1,3,3)
            ax3.set_title('GT')
            plt.imshow(gt)
            # plt.show()
            fig.savefig(f"./tmp/{filepath}/result", dpi=fig.dpi, bbox_inches='tight')

            all_preds.append(pred)
            all_gts.append(gt_e)

            clear()
            # Compute some metrics
            
            metrics(pred.ravel(), gt_e.ravel(), label_values=self.params['classes'], filepath=filepath)
            accuracy = metrics(
                predictions=np.concatenate([p.ravel() for p in all_preds]), 
                gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
                label_values=self.params['classes'],
                all=True,
                filepath=filepath
            )
            np.savetxt(f'./tmp/{filepath}/{filepath}_outp.csv', gt_e, delimiter=',')
            np.savetxt(f'./tmp/{filepath}/{filepath}_pred.csv', pred, delimiter=',')
        if all:
            return accuracy, all_preds, all_gts
        else:
            return accuracy
        

    def train(self):
        running_loss = 0.0
        
        if self.scheduler is not None:
            epoch = self.scheduler.last_epoch
        else:
            epoch = self.last_epoch + 1
        
        # epoch = 1 if epoch == 0 else epoch
        self.last_epoch = epoch
        
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

        self.net.train()
        for batch_idx, (data, target) in enumerate(self.loader['train']):
            # print('Training epoch {}/{}, Batch {}/{}, Iteração {}'.format(epoch, self.params['maximum_epochs'], batch_idx, len(self.loader['train']), self.iter_))
            data, target = self.prepare([data, target]) # Prepare input and labels 
            # data, target = Variable(data.cuda()), Variable(target.cuda())
            # data = data.to(self.params["device"], non_blocking=True)
            # target = target.to(self.params["device"], non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.net(data)
            # loss = CrossEntropy2d(output, target, self.weight_cls[0]) # Calculate the loss function
            dice_loss = DiceLoss(mode='multiclass', classes=self.weight_cls[0])(output, target)
            focal_loss = FocalLoss(mode='multiclass', gamma=2.0)(output, target)
            loss = dice_loss + (1*focal_loss)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            self.losses[self.iter_] = loss.item()
            self.mean_losses[self.iter_] = np.mean(self.losses[max(0,self.iter_-100):self.iter_])
            
            clear()
            print('Training (epoch {}/{}) [{}/{} ({:.0f}%)]\tIteração {}\tLoss: {:.6f}'.format(
                    epoch, self.params['maximum_epochs'], batch_idx, len(self.loader['train']),
                    100. * batch_idx / len(self.loader['train']), self.iter_, loss.item()
                )
            )
            
            if self.iter_ % self.print_each == 0:
                clear()
                image = data.data.cpu().numpy()[0]
                # print(image.shape)
                # rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                rgb = np.asarray(255 * np.transpose(image,(1,2,0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(epoch, self.params['maximum_epochs'], batch_idx, len(self.loader['train']),100. * batch_idx / len(self.loader['train']), loss.item(), accuracy(pred, gt)))
                fig = plt.figure()
                plt.plot(self.mean_losses[:self.iter_]) #and plt.show()
                fig.savefig(f"./tmp/train_mean_loss", dpi=fig.dpi, bbox_inches='tight')

                fig = plt.figure()
                ax1 = fig.add_subplot(131)
                ax1.set_title('RGB')
                plt.imshow(rgb)
                ax2 = fig.add_subplot(132)
                ax2.set_title('Ground truth')
                plt.imshow(convert_to_color(gt))
                ax3 = fig.add_subplot(133)
                ax3.set_title('Prediction')
                plt.imshow(convert_to_color(pred))
                # plt.show()
                fig.savefig(f"./tmp/train_progress", dpi=fig.dpi, bbox_inches='tight')
                
                self.last_loss = running_loss
                running_loss = 0.0
                
            self.iter_ += 1
        

            del(data, target, loss)
        
        # if epoch % self.params['save_epoch'] == 0:
        #     # We validate with the largest possible stride for faster computing
        #     acc = self.test(stride = min(self.params['window_size']), all=False)
            
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': self.net.state_dict(),
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #         # 'loss': loss,
        #     }, './segnet256_epoch{}_{:.2f}'.format(epoch, acc))
        #     # torch.save(net.state_dict(), './segnet256_epoch{}_{}'.format(e, acc))
        
        return self.last_loss
        
    # torch.save(net.state_dict(), './segnet_final')
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': net.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     # 'loss': loss,
    # }, './segnet_final')