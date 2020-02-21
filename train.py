import numpy as np
import torch
from torch import nn, optim

import params
from utils import make_tensor, cuda
from mnist import get_mnist


def train_model( model, data_loader, dump_location, log_dump_location ):

    optimizer = optim.Adam( model.parameters() )
                                
    criterion = nn.CrossEntropyLoss()

    log = []

    for epoch in range( params.num_epochs ):
        model.train()
        for step, ( images, lables ) in enumerate( data_loader ):
            
            if(params.gpu_flag):
                images = images.cuda(params.gpu_name)
                lables = lables.cuda(params.gpu_name)
            optimizer.zero_grad()

            preds = model( images )
            loss = criterion( preds, lables )
    
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(data_loader),
                              loss.data.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step == 0):
            epoch_loss, epoch_accuracy, _ = eval_model(model, data_loader)
            log.append([epoch,epoch_loss, epoch_accuracy])



    # # save final model
    torch.save(model, dump_location)
    np.savetxt(log_dump_location, np.array(log))
    return model



def eval_model( model, data_loader ):

    loss = 0
    accuracy = 0 

    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    predicted_array = []
    for (images, labels) in data_loader:

        if(params.gpu_flag):
            images = images.cuda(params.gpu_name)
            labels = labels.cuda(params.gpu_name)
        with torch.no_grad():
            preds = model(images)
        loss += criterion( preds, labels ).item()

        _, predicted = torch.max(preds.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted = predicted.detach().cpu()
        for i in predicted:
            predicted_array.append(i)

        # pred_cls = preds.data.max(1)[1]
        # print(pred_cls.eq(labels.data).cpu().sum())
        # accuracy += pred_cls.eq(labels.data).cpu().sum() / len(labels)

    
    loss /= len(data_loader)
    # accuracy /= len( data_loader )
    accuracy = correct/total

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, accuracy))

    return loss, accuracy, np.array(predicted_array)


