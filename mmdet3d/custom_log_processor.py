from mmengine.runner.log_processor import LogProcessor

class CustomLogProcessor(LogProcessor):
    def get_log_after_iter(self, runner, batch_idx, mode):
        log_str = super().get_log_after_iter(runner, batch_idx, mode)
        log_vars = runner.message_hub.get_info('log_vars')
        print(f"[CustomLogProcessor] log_vars: {log_vars}")
        if log_vars is not None:
            if mode == 'train':
                log_items = [
                    f"Epoch [{runner.epoch + 1}][{batch_idx + 1}/{len(runner.train_dataloader)}]",
                    f"lr: {log_vars.get('lr', 'NA')}",
                    f"data_time: {log_vars.get('data_time', 'NA')}",
                    f"iter_time: {log_vars.get('time', 'NA')}",
                    f"memory: {log_vars.get('memory', 'NA')}",
                    f"loss: {log_vars.get('loss', 'NA')}",
                    f"loss_cls: {log_vars.get('loss_cls', 'NA')}",
                    f"loss_bbox: {log_vars.get('loss_bbox', 'NA')}",
                    f"grad_norm: {log_vars.get('grad_norm', 'NA')}",
                ]
                print("[CustomLogProcessor][train] " + "  ".join(log_items))
            elif mode == 'val':
                # Add more validation metrics here as needed
                log_items = [
                    f"Epoch(val) [{runner.epoch + 1}][{batch_idx + 1}/{len(runner.val_dataloader)}]",
                    f"eta: {log_vars.get('eta', 'NA')}",
                    f"time: {log_vars.get('time', 'NA')}",
                    f"data_time: {log_vars.get('data_time', 'NA')}",
                    f"memory: {log_vars.get('memory', 'NA')}",
                    f"val_loss: {log_vars.get('loss', 'NA')}",
                    f"val_mAP: {log_vars.get('mAP', 'NA')}",
                    # Add more metrics as available
                ]
                print("[CustomLogProcessor][val] " + "  ".join(log_items))
        return log_str
