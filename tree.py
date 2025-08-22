import os

def print_tree(root_dir, exclude_dirs=None, prefix=""):
    """
    폴더 트리 구조 출력 (제외 폴더 지원)

    :param root_dir: 탐색 시작 경로
    :param exclude_dirs: 제외할 폴더 이름 리스트
    :param prefix: 출력 시 들여쓰기용
    """
    if exclude_dirs is None:
        exclude_dirs = []

    # 현재 디렉터리의 항목 정렬
    entries = sorted(os.listdir(root_dir))
    entries = [e for e in entries if e not in exclude_dirs]

    for index, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if index == len(entries) - 1 else "│   "
            print_tree(path, exclude_dirs, prefix + extension)


if __name__ == "__main__":
    root_path = "D:\dev\django\BerthDemo"  # 현재 폴더부터 시작
    exclude = [".venv", ".idea", ".git","__pycache__","create.txt"]  # 제외할 폴더
    print_tree(root_path, exclude)
