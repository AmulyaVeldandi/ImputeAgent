# Impute-Agent CDK

Deploys S3 (raw+outputs), Lambda (S3+API trigger), API Gateway, and IAM permissions
for Bedrock and SageMaker. Configure context in `cdk.json` then:

```bash
npm install
cdk bootstrap
cdk deploy
```
