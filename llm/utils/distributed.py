class WaitForMainProcess:
    """
    Context manager to avoid duplicate computation across processes, e.g. data loading/preprocessing.
    Computation must store state to disk, which can be loaded later by other processes (e.g. cached datasets).
    """

    def __init__(self, accelerator):
        self.accelerator = accelerator

    def __enter__(self):
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()

    def __exit__(self, *_):
        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
