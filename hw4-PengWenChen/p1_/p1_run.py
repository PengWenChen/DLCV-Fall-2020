import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import csv
import pdb
import os

from .utils import MiniDataset, worker_init_fn, GeneratorSampler, euclidean_metric, count_acc, cosine_sim
from .model import Convnet, Parametric

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

class Prototypical():
    def __init__(self, args):
        self.args = args
        self.lr = 1e-3
        self.max_epoch = 50

        self.train_classes = 64
        self.train_img_number_per_class = 600
        self.train_episode = 2400 # 2400, 2000
        self.train_N_way = 10 # 10, 5
        self.train_N_shot = 1
        self.train_N_query = 7 # 7, 15

        self.test_N_way = 5
        self.test_N_shot = 1
        self.test_N_query = 15

        self.similarity = args.metric # "cosine, parametric"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_loader(self, csv, img_dir, case_csv, N_way, N_query, N_shot):
        dataset = MiniDataset(csv, img_dir)
        data_loader = DataLoader(
            dataset, batch_size=N_way * (N_query + N_shot),
            num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
            sampler=GeneratorSampler(case_csv))
        return dataset, data_loader

    def preprocess(self):
        class_list = np.arange(self.train_classes) # 0~63
        img_list = np.arange(self.train_img_number_per_class) # 0~599
        support_gt = np.arange(10).tolist()
        csv_query_gt = []

        with open('./hw4_data/train_case_p1_3_5way10shot.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            first_row = ['episode_id']
            # for shot_num in range(self.train_N_shot):
            #     for i in range(self.train_N_way):
            #         first_row.append(f'class{i}_support{shot_num}')
            for i in range(self.train_N_way):
                for shot_num in range(self.train_N_shot):
                    first_row.append(f'class{i}_support{shot_num}')
            for i in range(self.train_N_query*self.train_N_way):
                first_row.append(f'query{i}')
            writer.writerow(first_row)

            for train_e in range(self.train_episode):
                support_img_id = []
                print(f"train case: {self.train_episode}", end='\r')
                np.random.shuffle(class_list)
                np.random.shuffle(img_list)

                support_class = ( class_list[:self.train_N_way].tolist() )
                # for shot_num in range(self.train_N_shot):
                #     support_img_id = support_img_id + (self.train_img_number_per_class*np.array(support_class)+img_list[shot_num]).tolist()
                for way in range(self.train_N_way):
                    for shot_num in range(self.train_N_shot):
                        support_img_id.append(self.train_img_number_per_class*support_class[way]+img_list[shot_num])
                row_list = [train_e] + support_img_id

                query_img_id = []
                query_gt = []
                for j in range(self.train_N_query):
                    for k in range(len(support_class)):
                        query_gt.append(k)
                        query_img_id.append( self.train_img_number_per_class * support_class[k] + img_list[self.train_N_shot+j] )
                
                c = list(zip(query_gt, query_img_id))
                random.shuffle(c)
                query_gt, query_img_id = zip(*c)
                query_gt = list(query_gt)
                query_img_id = list(query_img_id)

                row_list = row_list + query_img_id
                csv_query_gt.append([train_e] + query_gt)
                writer.writerow(row_list)
        
        with open('./hw4_data/train_case_gt_p1_3_5way10shot.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            first_row = ['episode_id']
            for i in range(self.train_N_query*self.train_N_way):
                first_row.append(f'query{i}')
            writer.writerow(first_row)
            for i in range(len(csv_query_gt)):
                print(f"gt: {i}", end='\r')
                writer.writerow(csv_query_gt[i])

    def test(self):
        valid_set, valid_loader = self.build_loader(
                                        self.args.test_img, 
                                        self.args.test_img_dir, 
                                        self.args.test_case,
                                        self.test_N_way,
                                        self.test_N_query,
                                        self.test_N_shot)

        model = Convnet()
        # self.load_checkpoint('model/best_p1_1.pth', model)
        self.load_checkpoint(self.args.model, model)
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        val_acc = []
        val_loss = []
        prediction_results = []

        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(self.device)

            support_input = data[:self.test_N_way * self.test_N_shot,:,:,:] 
            query_input   = data[self.test_N_way * self.test_N_shot:,:,:,:]

            if self.similarity=="parametric":
                proto = model(support_input, query_input)
            else:
                proto = model(support_input)

            label_encoder = {target[i * self.test_N_shot] : i for i in range(self.test_N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.test_N_way * self.test_N_shot:]])

            if self.similarity=="euclidean":
                logits = euclidean_metric(model(query_input), proto)
            elif self.similarity=="cosine":
                logits = cosine_sim(model(query_input), proto)
            elif self.similarity=="parametric":
                logits = proto
                
            loss = criterion(logits, query_label)
            acc = count_acc(logits, query_label)

            val_loss.append(loss.item())
            val_acc.append(acc)

            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            pred = [batch_idx] + pred
            prediction_results.append(pred)
            proto = None; logits = None; loss = None
        print('val_loss: {}, val_acc: {}\n'.format(np.mean(val_loss), np.mean(val_acc)))
        
        with open(self.args.test_case_output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            first_row = ['episode_id']
            
            for i in range(self.test_N_query * self.test_N_way):
                first_row.append(f'query{i}')
            writer.writerow(first_row)

            writer.writerows(prediction_results)
        
        print('finish')

    def train(self):
        train_set, train_loader = self.build_loader(
                                        self.args.train_img, 
                                        self.args.train_img_dir, 
                                        self.args.train_case,
                                        self.train_N_way,
                                        self.train_N_query,
                                        self.train_N_shot)

        valid_set, valid_loader = self.build_loader(
                                        self.args.test_img, 
                                        self.args.test_img_dir, 
                                        self.args.test_case,
                                        self.test_N_way,
                                        self.test_N_query,
                                        self.test_N_shot)

        if self.similarity=="parametric":
            model = Parametric(self.test_N_way)
        else:
            model = Convnet()
        
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        iteration = 0
        best_acc = 0
        for epoch in range(self.max_epoch):
            
            model.train()
            train_acc = []
            train_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)

                support_input = data[:self.train_N_way * self.train_N_shot,:,:,:] 
                query_input   = data[self.train_N_way * self.train_N_shot:,:,:,:]

                if self.similarity=="parametric":
                    proto = model(support_input, query_input)
                else:
                    proto = model(support_input).reshape(self.train_N_shot, self.train_N_way, -1).mean(dim=0)

                label_encoder = {target[i * self.train_N_shot] : i for i in range(self.train_N_way)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.train_N_way * self.train_N_shot:]])

                if self.similarity=="euclidean":
                    logits = euclidean_metric(model(query_input), proto)
                elif self.similarity=="cosine":
                    # pdb.set_trace()
                    logits = cosine_sim(model(query_input), proto)
                elif self.similarity=="parametric":
                    logits = proto

                # pdb.set_trace()
                loss = criterion(logits, query_label)
                acc = count_acc(logits, query_label)

                if iteration % 500 == 0:
                    print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                        .format(epoch, batch_idx, len(train_loader), loss.item(), acc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_acc.append(acc)
                train_loss.append(loss.item())
                
                iteration += 1
                proto = None; logits = None; loss = None

            lr_scheduler.step()

            print('\nEpoch: {}, train_loss: {}, train_acc: {}'.format(epoch, np.mean(train_loss), np.mean(train_acc)))

            model.eval()
            val_acc = []
            val_loss = []
            for batch_idx, (data, target) in enumerate(valid_loader):
                data = data.to(self.device)

                support_input = data[:self.test_N_way * self.test_N_shot,:,:,:] 
                query_input   = data[self.test_N_way * self.test_N_shot:,:,:,:]

                if self.similarity=="parametric":
                    proto = model(support_input, query_input)
                else:
                    proto = model(support_input)

                label_encoder = {target[i * self.test_N_shot] : i for i in range(self.test_N_way)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.test_N_way * self.test_N_shot:]])

                if self.similarity=="euclidean":
                    logits = euclidean_metric(model(query_input), proto)
                elif self.similarity=="cosine":
                    logits = cosine_sim(model(query_input), proto)
                elif self.similarity=="parametric":
                    logits = proto
                    
                loss = criterion(logits, query_label)
                acc = count_acc(logits, query_label)

                val_loss.append(loss.item())
                val_acc.append(acc)

                proto = None; logits = None; loss = None
            print('Epoch: {}, val_loss: {}, val_acc: {}\n'.format(epoch, np.mean(val_loss), np.mean(val_acc)))

            if best_acc < np.mean(val_acc):
                self.save_checkpoint(os.path.join(self.args.model_save_dir, 'best_p1_1_v2.pth'), model, optimizer)

    def save_checkpoint(self, checkpoint_path, model, optimizer):
        print('\nStart saving ...')
        state = {'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=False):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        print('model loaded from %s\n' % checkpoint_path)





        