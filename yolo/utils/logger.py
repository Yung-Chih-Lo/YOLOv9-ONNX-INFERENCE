import logging
import os
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Logger 類使用了單例模式，確保在應用程序中只會有一個 Logger 實例。
        這是通過在 __new__ 方法中檢查 _instance 是否為 None 來實現的。
        如果 _instance 為 None，則創建一個新的實例，否則返回已有的實例。

        Returns:
            _type_: _description_
        """
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir="logs", log_file=None):
        if self._initialized:
            return
        self._initialized = True

        if log_file is None:
            log_file = f"log_{datetime.now().strftime('%Y%m%d_%H')}.log"
        
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(self.log_dir, self.log_file)

        # 創建日誌目錄（如果不存在）
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 設置全局的日誌配置
        self.configure_global_logging()

    def configure_global_logging(self):
        # 創建處理器 - 文件處理器
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # 設置格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 獲取全局的 logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 添加處理器到全局 logger
        root_logger.addHandler(file_handler)
        
        # 防止重複添加處理器
        root_logger.propagate = False

    def get_logger(self):
        return logging.getLogger("custom_logger")

# 示例主程序
if __name__ == "__main__":
    # 初始化 Logger
    Logger()

    # 使用全局 logger 進行日誌記錄
    logging.debug("這是一條調試訊息")
    logging.info("這是一條資訊訊息")
    logging.warning("這是一條警告訊息")
    logging.error("這是一條錯誤訊息")
    logging.critical("這是一條嚴重錯誤訊息")
