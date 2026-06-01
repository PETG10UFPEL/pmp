import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# No Windows local, evita conflito com a pasta .env (venv)
if Path('.env.local').exists():
    load_dotenv('.env.local', override=True)
else:
    load_dotenv(override=True)

SCOPES = [
    'https://www.googleapis.com/auth/drive',
]
FOLDER_MIME = 'application/vnd.google-apps.folder'
INDEX_ZIP_NAME = 'chroma_index.zip'

EXCLUDED_DIR_NAMES = {
    '_chroma_index',
    'indice_backup',
    'index_backup',
    'chroma_db',
    'chroma_index',
    '__pycache__',
}
ALLOWED_EXTENSIONS = {'.pdf'}


def get_drive_service():
    creds = None

    try:
        if 'gcp_service_account' in st.secrets:
            info = st.secrets['gcp_service_account']
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass

    if not creds:
        json_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'disco-retina-371501-525385725005.json')
        if os.path.exists(json_path):
            creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)

    if not creds:
        raise RuntimeError(
            "Credenciais do Google Cloud não encontradas. "
            "Configure 'gcp_service_account' nos Secrets do Streamlit ou tenha o arquivo JSON localmente."
        )

    return build('drive', 'v3', credentials=creds)


def _list_children(service, folder_id: str):
    q = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            fields='nextPageToken, files(id, name, mimeType, size)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return items


def _download_file(service, file_id: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _is_excluded_dir_name(name: str) -> bool:
    return name.strip().lower() in EXCLUDED_DIR_NAMES


def _is_allowed_file(name: str, mime: str) -> bool:
    if mime.startswith('application/vnd.google-apps'):
        return False
    suffix = Path(name).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def _sync_folder_recursive(service, folder_id: str, out_root: Path, rel: Path = Path('.')) -> list[Path]:
    downloaded: list[Path] = []
    items = _list_children(service, folder_id)

    for it in items:
        name = it['name']
        mime = it.get('mimeType', '')
        it_id = it['id']

        if mime == FOLDER_MIME:
            if _is_excluded_dir_name(name):
                print(f'[SYNC] Ignorando subpasta excluída: {rel / name}')
                continue
            downloaded.extend(_sync_folder_recursive(service, it_id, out_root, rel / name))
            continue

        if not _is_allowed_file(name, mime):
            print(f'[SYNC] Ignorando arquivo não suportado: {rel / name}')
            continue

        dest = out_root / rel / name
        if not dest.exists():
            _download_file(service, it_id, dest)
            downloaded.append(dest)

    return downloaded


def sync_folder(folder_id: str, out_dir: str, recursive: bool = True) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        service = get_drive_service()

        if recursive:
            downloaded = _sync_folder_recursive(service, folder_id, out, Path('.'))
        else:
            downloaded = []
            items = _list_children(service, folder_id)
            for f in items:
                name = f['name']
                file_id = f['id']
                mime = f.get('mimeType', '')

                if mime == FOLDER_MIME:
                    if _is_excluded_dir_name(name):
                        print(f'[SYNC] Ignorando subpasta excluída: {name}')
                    continue

                if not _is_allowed_file(name, mime):
                    print(f'[SYNC] Ignorando arquivo não suportado: {name}')
                    continue

                dest = out / name
                if not dest.exists():
                    _download_file(service, file_id, dest)
                    downloaded.append(dest)

        print(f'Baixados {len(downloaded)} arquivos para {out.resolve()}')
        return downloaded

    except Exception as e:
        print(f'Erro na sincronização: {e}')
        return []


def upload_index_to_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    import tempfile
    import zipfile

    # Garante sempre /tmp/chroma_db na nuvem
    if Path("/mount/src").exists():
        db_dir = "/tmp/chroma_db"

    db_path = Path(db_dir)
    if not db_path.exists():
        print(f'[ERRO] Pasta do índice não existe: {db_path}')
        return False

    files_to_upload = [f for f in db_path.rglob('*') if f.is_file()]
    if not files_to_upload:
        print('[AVISO] Nenhum arquivo encontrado para upload.')
        return False

    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_upload:
                zf.write(file_path, file_path.relative_to(db_path))

        print(f'[Drive] ZIP criado: {zip_path.stat().st_size // 1024} KB')

        service = get_drive_service()

        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_ZIP_NAME}' "
            f"and trashed = false"
        )
        resp = service.files().list(
            q=q,
            fields='files(id)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        existing = resp.get('files', [])

        media = MediaFileUpload(str(zip_path), mimetype='application/zip', resumable=False)

        if existing:
            service.files().update(
                fileId=existing[0]['id'],
                media_body=media,
                supportsAllDrives=True,
            ).execute()
            print(f"[Drive] ZIP atualizado: '{INDEX_ZIP_NAME}'.")
        else:
            service.files().create(
                body={'name': INDEX_ZIP_NAME, 'parents': [gdrive_folder_id]},
                media_body=media,
                fields='id',
                supportsAllDrives=True,
            ).execute()
            print(f"[Drive] ZIP criado no Drive: '{INDEX_ZIP_NAME}'.")

        zip_path.unlink(missing_ok=True)
        return True

    except Exception as e:
        print(f'[ERRO] Falha ao salvar índice no Drive: {e}')
        return False


def download_index_from_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    import tempfile
    import zipfile

    try:
        service = get_drive_service()

        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_ZIP_NAME}' "
            f"and trashed = false"
        )
        resp = service.files().list(
            q=q,
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get('files', [])

        if not files:
            print(f"[Drive] Arquivo '{INDEX_ZIP_NAME}' não encontrado no Drive.")
            return False

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)

        _download_file(service, files[0]['id'], zip_path)
        print(f'[Drive] ZIP baixado: {zip_path.stat().st_size // 1024} KB')

        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(db_path)

        zip_path.unlink(missing_ok=True)
        print(f"[Drive] Índice extraído para '{db_path}'.")
        return True

    except Exception as e:
        print(f'[ERRO] Falha ao baixar índice do Drive: {e}')
        return False


def index_exists_on_drive(gdrive_folder_id: str) -> bool:
    try:
        service = get_drive_service()
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_ZIP_NAME}' "
            f"and trashed = false"
        )
        resp = service.files().list(
            q=q,
            fields='files(id)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return len(resp.get('files', [])) > 0
    except Exception:
        return False


if __name__ == '__main__':
    pass
