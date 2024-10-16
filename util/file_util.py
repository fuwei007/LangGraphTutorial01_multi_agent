import os
import uuid
from datetime import datetime

class FileUtils:
    @staticmethod
    def upload_file(file: bytes, file_path: str, file_name: str) -> None:
        """文件上传"""
        target_file = os.path.dirname(file_path)
        if not os.path.exists(target_file):
            os.makedirs(target_file)
        with open(os.path.join(file_path, file_name), 'wb') as out:
            out.write(file)

    @staticmethod
    def get_folder() -> str:
        """上传文件夹以 yyyy-MM-dd 命名"""
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def random_uuid() -> str:
        """产生一个36个字符的UUID"""
        return str(uuid.uuid4())
