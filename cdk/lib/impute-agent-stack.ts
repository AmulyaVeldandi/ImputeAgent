import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as path from 'path';

interface StackProps extends cdk.StackProps {
  bedrockRegion: string;
  sagemakerEndpointName: string;
}

export class ImputeAgentStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: StackProps) {
    super(scope, id, props);

    const rawBucket = new s3.Bucket(this, 'RawBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true
    });

    const outBucket = new s3.Bucket(this, 'OutputsBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true
    });

    const fn = new lambda.Function(this, 'ImputeAgentLambda', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'lambda_handler.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../lambda')),
      memorySize: 1024,
      timeout: cdk.Duration.seconds(60),
      environment: {
        RAW_BUCKET: rawBucket.bucketName,
        OUTPUTS_BUCKET: outBucket.bucketName,
        BEDROCK_REGION: props.bedrockRegion,
        SAGEMAKER_ENDPOINT: props.sagemakerEndpointName,
        LLM_IMPUTER_ENABLED: 'true'
      }
    });

    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['logs:CreateLogGroup','logs:CreateLogStream','logs:PutLogEvents'],
      resources: ['*']
    }));

    rawBucket.grantRead(fn);
    outBucket.grantReadWrite(fn);

    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:InvokeEndpoint'],
      resources: ['*']
    }));

    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock:InvokeModel','bedrock:InvokeModelWithResponseStream'],
      resources: ['*']
    }));

    rawBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED_PUT,
      new s3n.LambdaDestination(fn)
    );

    const api = new apigw.LambdaRestApi(this, 'ImputeApi', {
      handler: fn,
      proxy: false,
      deployOptions: { stageName: 'prod' }
    });

    const impute = api.root.addResource('impute');
    impute.addMethod('POST');

    new cdk.CfnOutput(this, 'ApiUrl', { value: api.urlForPath('/impute') });
    new cdk.CfnOutput(this, 'RawBucketName', { value: rawBucket.bucketName });
    new cdk.CfnOutput(this, 'OutputsBucketName', { value: outBucket.bucketName });
  }
}
