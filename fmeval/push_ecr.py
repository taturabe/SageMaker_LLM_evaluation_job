import boto3
import subprocess

account_id = boto3.client('sts').get_caller_identity().get('Account')
region = boto3.Session().region_name
ecr_repository = 'fmeval-processing-container'
tag = ':latest'
processing_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account_id, region, ecr_repository + tag)


# Dockerイメージのビルド
subprocess.run(f"docker build -t {ecr_repository} ./", shell=True, check=True)

# ECRへのログイン
subprocess.run(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com", shell=True, check=True)

# ECRリポジトリの作成
subprocess.run(f"aws ecr create-repository --repository-name {ecr_repository}", shell=True, check=True)

# Dockerイメージのタグ付け
subprocess.run(f"docker tag {ecr_repository}{tag} {processing_repository_uri}", shell=True, check=True)

# DockerイメージをECRにプッシュ
subprocess.run(f"docker push {processing_repository_uri}", shell=True, check=True)
