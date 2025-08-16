---
author: [""]
title: "Authenticating AWS with EKS"
date: "2025-08-16"
description: "Different ways to allow EKS workloads to securely access AWS services"
tags: ["k8s", "eks", "aws"]
summary: "How to authenticate EKS workloads with AWS services?"
ShowToc: false
ShowBreadCrumbs: false
---

If you are using AWS SDK or AWS CLI to make API requests to AWS services from your application, it relies on a **credential chain** to determine whether you have sufficient permissions.  

The [default credential chain](https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html#credentialProviderChain) uses the exported AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` or `AWS_REGION`) environment variables or checks for these values under `$HOME/.aws/credentials` file.

When developing locally, it’s easy to store credentials in your `$HOME/.aws/credentials` file or export them in environment variables. This works well for local testing, but once you deploy to Kubernetes (or Amazon EKS), you need a secure and automated way for pods to obtain AWS credentials **without hardcoding secrets**.

## Three common ways for EKS workloads to authenticate to AWS

### Node IAM Role

When you create an EKS cluster with EC2 worker nodes (managed node groups), each EC2 instance is launched with an instance profile. An instance profile contains the role and allows the application running on EC2 instance to get temporary credentials. Instead of storing AWS keys in the pod, containers running on that node obtain AWS credentials by calling the EC2 Instance Metadata Service (IMDS), which returns [temporary credentials](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html#instance-metadata-security-credentials) tied to the instance profile.

To grant permissions (for example, to pull from ECR or access S3), you create an IAM policy with the required permissions and attach it to the node instance role. Any pod running on that node can then use those permissions, which is simple.

The disadvantage of this approach is that because the node role is shared by all pods on that node, it breaks the principle of least privilege.

### IAM Roles for Service Accounts (IRSA)

Similar to how EC2 instance profiles provide credentials to EC2 instances, IRSA uses a Kubernetes [service account](https://kubernetes.io/docs/concepts/security/service-accounts/) to get temporary credentials. This service account is associated with an IAM role, and any application that requires access to an AWS service assumes the role using this service account.

There is a detailed blog [IAM roles for Kubernetes service accounts - deep dive](https://mjarosie.github.io/dev/2021/09/15/iam-roles-for-kubernetes-service-accounts-deep-dive.html) by Maciej that explains IRSA better than I could do justice. In short,

* An OIDC identity provider is setup for the EKS cluster.
* An IAM role is created whose trust policy allows `AssumeRoleWithWebIdentity` from that OIDC provider and the specific `system:serviceaccount:namespace:name` subject.
* A Kubernetes ServiceAccount is created or annotated with the role ARN.
* Pods using that ServiceAccount get a projected token, which the AWS SDK exchanges for temporary credentials. The AWS secrets are injected and mounted by [Amazon EKS Pod Identity Webhook](https://github.com/aws/amazon-eks-pod-identity-webhook).

The disadvantage is that this process is more involved compared to other options.

### EKS Pod Identity (Recommended)

Similar to IRSA, EKS Pod Identity also maps a role to a service account but the process is much simpler. In this approach, EKS starts a Pod Identity Agent that runs as a pod on each node to handle credential issuance and associations. An EKS Pod Identity association is created to bind a Kubernetes service account to an IAM role. Any pod that uses this service account can access the AWS services that the role has permission for.

The advantage of this approach is that it is simpler than IRSA and works across clusters without editing role trust policies for each cluster, as IRSA might require. It only requires the Pod Identity Agent to be deployed on every node.

The recommended way to provision EKS Pod Identity Agent is to use it as an EKS Add-on. The pod identity association can be created either using AWS EKS console or AWS cli [`create-pod-identity-association`](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html) or using the [`aws_eks_pod_identity_association`](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/eks_pod_identity_association) resource in terraform.
