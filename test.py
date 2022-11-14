import numpy as np
import os
from configargparse import ArgParser
import torch
import torch.utils.data as Data
from tqdm import tqdm
from util.Evaluation import myMSE
from data_processing.division import get_division_len
from model.model import DI_ConvGRU
from train import train_valid_seq_split

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def create_predictions(model, train_loader):
    """Create  predictions"""
    finalpredict=[]
    for step, (data1, data2, label) in tqdm(enumerate(train_loader)):
        #ERA5_validation = torch.as_tensor(Variable(data1), dtype=torch.float).cuda()
        data1 = torch.as_tensor(data1, dtype=torch.float).cuda()
        data2 = torch.as_tensor(data2, dtype=torch.float).cuda()
        with torch.no_grad():
            last_state = model(data1, data2)
            last_state = torch.squeeze(last_state)
            label = torch.squeeze(label)
            pred = last_state.cpu().detach().numpy()
            if step == 0:
                finalpredict = pred
                obs = label
            else:
                finalpredict = np.concatenate((finalpredict, pred), axis=0)
                obs = np.concatenate((obs, label), axis=0)
    return finalpredict, obs


def main(division_size, division_bias, batch_size):
    lon = np.load('./data/lon.npy')
    lat = np.load('./data/lat.npy')
    width, height, widtha, heightb, widtha_down, heightb_rigint \
        = get_division_len(lon, lat, division_size, division_bias)

    fill_prediction = np.zeros((365,102,167))
    total_predictionmask = np.zeros((102,167))

    count = 0
    division_len = (len(widtha) * len(heightb)) + (len(widtha_down) * len(heightb_rigint))
    for i in range(division_len):
        dir = r"./data/data_division" + str(count)
        dir1 = dir + '/data_model_lag3/'
        ERA5_test = np.load(dir1 + 'ERA5_test.npy')
        ERA5_test = ERA5_test[:, :, 3, :, :]
        ERA5_test = ERA5_test[:, :, np.newaxis, :, :]
        #static_test = np.load(dir1 + 'static_test.npy')
        # static_test = static_test[:, :, 0:4, :, :]
        #features_test = np.concatenate((ERA5_test, static_test), axis=2)
        sm_test = np.load(dir1 + 'sm_test.npy')
        label_test = np.load(dir1 + 'label_test.npy')
        division_mask = np.load(dir + '/division_nmask.npy')
        fill_prediction1 = np.zeros((365,102,167))
        label_test2 = label_test.reshape(label_test.shape[0]*label_test.shape[1]*label_test.shape[2],
        label_test.shape[3])
        if np.min(label_test2) != np.nan:
            train_loader = train_valid_seq_split(ERA5_test, sm_test, label_test, batch_size, shuffle=False)
            sample, time_step, features, lons, lats = ERA5_test.shape
            model = DI_ConvGRU(input_size=(lons, lats), input_dim=2,
                                hidden_dim=[64, 64, 1], kernel_size=(3, 3),
                                num_layers=3)
            model.cuda()
            model.load_state_dict(torch.load(dir1 + 'model_tp.pth'),False)

            pred, obs = create_predictions(model, train_loader)
            print('finished testing data' + str(count + 1) + 'and' + str(division_len - count - 1) + ' left')
            ### ############测试单独区域性能
            # 计算RMSES
            result = myMSE(pred, obs)
        else:
            pred = np.zeros(label_test.shape)
        count = count + 1


        division_mas_height, division_mas_width = np.where(division_mask == 1)
        fill_prediction1[:, division_mas_height.min():division_mas_height.max()+1
        , division_mas_width.min():division_mas_width.max()+1] = pred[:,0:pred.shape[1]+1,0:pred.shape[2]+1]

        fill_prediction = fill_prediction + fill_prediction1

        total_predictionmask = total_predictionmask + division_mask

    fill_prediction = fill_prediction / total_predictionmask
    print(fill_prediction.shape)
    print('end')
    dir1 = r"./results/DI_model_lag3"
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    np.save(dir1 + '/pred_tp.npy', fill_prediction)

    # TODO: Computer score of R2 and RMSE
    mask = np.load('./data/mask.npy')
    # myMSE(fill_prediction, fill_label_test)


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--division_size', type=int, default=50, help='division size of small input for training the model')
    p.add_argument('--division_bias', type=int, default=8, help='division bias of small input for training the model')
    p.add_argument('--batch_size', type=int, default=16, help='batch_size')
    args = p.parse_args()

    main(
        division_size=args.division_size,
        division_bias=args.division_bias,
        batch_size=args.batch_size
    )
