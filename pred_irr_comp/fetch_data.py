import zipfile
from pathlib import Path
import boto3
from loguru import logger
from tqdm import tqdm
import typer
from pred_irr_comp.config import RAW_DATA_DIR

app = typer.Typer()

BUCKET_NAME = 'test-pred-irr'
AWS_PROFILE = 'imaad-airrigate-datascientist'


def _download_with_progress(s3, bucket, key, dest):
    meta = s3.head_object(Bucket=bucket, Key=key)
    total_size = meta['ContentLength']

    with tqdm(total=total_size, unit='B', unit_scale=True, desc=key) as pbar:
        s3.download_file(
            bucket,
            key,
            str(dest),
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )

def download_to_raw(output_dir: Path = RAW_DATA_DIR) -> None:
    """Download competition zip from S3 and extract into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / 'dataset.zip'

    logger.info(f'Downloading from s3://{BUCKET_NAME}/ ...')
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client('s3')

    objects = s3.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' not in objects:
        logger.error(f'No objects found in bucket {BUCKET_NAME}')
        raise typer.Exit(code=1)

    zip_keys = [obj['Key'] for obj in objects['Contents'] if obj['Key'].endswith('.zip')]
    if not zip_keys:
        logger.error('No .zip files found in the bucket')
        raise typer.Exit(code=1)

    zip_key = zip_keys[0]
    logger.info(f'Downloading {zip_key} to {zip_path}')
    _download_with_progress(s3, BUCKET_NAME, zip_key, zip_path)
    logger.success(f'Downloaded {zip_key}')

    logger.info(f'Extracting {zip_path} to {output_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for member in tqdm(members, desc='Extracting'):
            zf.extract(member, output_dir)
    logger.success(f'Extracted {len(members)} files to {output_dir}')

    zip_path.unlink()
    logger.info('Removed temporary zip file')


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
):
    download_to_raw(output_dir)


if __name__ == '__main__':
    app()
