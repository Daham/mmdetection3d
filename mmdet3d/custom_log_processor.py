from mmengine.runner.log_processor import LogProcessor

class CustomLogProcessor(LogProcessor):
    def get_log_after_iter(self, runner, batch_idx, mode):
        log_str = super().get_log_after_iter(runner, batch_idx, mode)
        # Add or format fields as needed
        log_vars = runner.message_hub.get_info('log_vars')
        # Compose your custom log string
        log_items = [
            f"Epoch [{runner.epoch + 1}][{batch_idx + 1}/{len(runner.train_dataloader)}]",
            f"lr: {log_vars.get('lr', 'NA'):.4e}",
            f"data_time: {log_vars.get('data_time', 'NA'):.4f}",
            f"iter_time: {log_vars.get('time', 'NA'):.4f}",
            f"memory: {log_vars.get('memory', 'NA')}",
            f"loss: {log_vars.get('loss', 'NA'):.4f}",
            f"loss_cls: {log_vars.get('loss_cls', 'NA'):.4f}",
            f"loss_bbox: {log_vars.get('loss_bbox', 'NA'):.4f}",
            f"grad_norm: {log_vars.get('grad_norm', 'NA'):.4f}",
        ]
        return "  ".join(log_items)
