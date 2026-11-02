"""下載固定版本的研究資料，排除作者的虛擬環境。"""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = 'ontoncom/Subtext-Decor-90'
REVISION = 'c24d01cc50384d9e01527bce002ceeeaf363ab32'
DESTINATION = Path(__file__).resolve().parent / 'data' / 'Subtext-Decor-90'

if __name__ == '__main__':
    print(snapshot_download(
        REPO_ID, repo_type='dataset', revision=REVISION,
        allow_patterns=['README.md', '.gitattributes', '[0-9]*/*.png'],
        local_dir=DESTINATION, max_workers=6,
    ))
