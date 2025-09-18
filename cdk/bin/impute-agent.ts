#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { ImputeAgentStack } from '../lib/impute-agent-stack';

const app = new cdk.App();
const bedrockRegion = app.node.tryGetContext('bedrockRegion') ?? 'us-east-1';
const sagemakerEndpointName = app.node.tryGetContext('sagemakerEndpointName') ?? 'impute-agent-endpoint';

new ImputeAgentStack(app, 'ImputeAgentStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
  },
  bedrockRegion,
  sagemakerEndpointName,
});
