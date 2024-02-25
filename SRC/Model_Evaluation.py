import Model_Training_Sequence

class Model_Evaluation:
    def __epoch_evaluate(Self, epoch, criterion):
       # Call during training of evaluation process
       with torch.no_grad():
           num_correct = 0
           num_total_images = 0
           running_loss = 0.0

           for images, targets in tqdm(self.validation_data_loader):
               with record_function("eval.to_device"):
                   images = images.to(self.device, non_blocking = self.dataloading_config.non_blocking)
                   one_hot_targets = targets.to(self.device, non_blocking = self.dataloading_config.non_blocking)
                
                with record_function("eval.forward"):
                   outputs = self.model(images)
                   loss = criterion(outputs, one_hot_targets)
                   running_loss += loss.item()*images.size(0)

                   correct = torch.argmax(outputs,dim=-1) == (targets.to(self.device))
                   num_correct += torch.sum(correct).item()
                   num_total_images += len(images)
        return running_loss, num_correct, num_total_images
    