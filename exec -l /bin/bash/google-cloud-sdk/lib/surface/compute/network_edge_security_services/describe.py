# -*- coding: utf-8 -*- #
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command for describing network edge security services."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.network_edge_security_services import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.network_edge_security_services import flags


class Describe(base.DescribeCommand):
  r"""Describe a Compute Engine network edge security service.

  *{command}* displays all data associated with a Compute Engine network edge
  security service in a project.

  ## EXAMPLES

  To describe a network edge security service with the name 'my-service' in
  region 'us-central1', run:

    $ {command} my-service \
      --region=us-central1
  """

  NETWORK_EDGE_SECURITY_SERVICE_ARG = None

  @classmethod
  def Args(cls, parser):
    cls.NETWORK_EDGE_SECURITY_SERVICE_ARG = (
        flags.NetworkEdgeSecurityServiceArgument())
    cls.NETWORK_EDGE_SECURITY_SERVICE_ARG.AddArgument(
        parser, operation_type='describe')

  def Run(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    ref = self.NETWORK_EDGE_SECURITY_SERVICE_ARG.ResolveAsResource(
        args, holder.resources)
    network_edge_security_service = client.NetworkEdgeSecurityService(
        ref, compute_client=holder.client)

    return network_edge_security_service.Describe()
