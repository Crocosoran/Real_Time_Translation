"""Generated client library for cloudkms version v1."""
# NOTE: This file is autogenerated and should not be edited by hand.
# NOTENOTE: This file has been gently hand-edited. Do not simply replace
# with an autogenerated version - merge with this version!
# For instance, the __future__ import immediately below...
from __future__ import absolute_import

import os
import platform
import sys

from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages

import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util


class CloudkmsV1(base_api.BaseApiClient):
  """Generated client library for service cloudkms version v1."""

  MESSAGES_MODULE = messages
  BASE_URL = u'https://cloudkms.googleapis.com/'

  _PACKAGE = u'cloudkms'
  _SCOPES = [u'https://www.googleapis.com/auth/cloud-platform']
  _VERSION = u'v1'
  _CLIENT_ID = 'nomatter'
  _CLIENT_SECRET = 'nomatter'
  _USER_AGENT = 'apitools gsutil/%s Python/%s (%s)' % (
      gslib.VERSION, platform.python_version(), sys.platform)
  if system_util.InvokedViaCloudSdk():
    _USER_AGENT += ' google-cloud-sdk'
    if system_util.CloudSdkVersion():
      _USER_AGENT += '/%s' % system_util.CloudSdkVersion()
  if MetricsCollector.IsDisabled():
    _USER_AGENT += ' analytics/disabled'
  else:
    _USER_AGENT += ' analytics/enabled'
  _CLIENT_CLASS_NAME = u'CloudkmsV1'
  _URL_VERSION = u'v1'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None):
    """Create a new cloudkms handle."""
    url = url or self.BASE_URL
    super(CloudkmsV1, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers)
    self.projects_locations_keyRings_cryptoKeys_cryptoKeyVersions = self.ProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsService(self)
    self.projects_locations_keyRings_cryptoKeys = self.ProjectsLocationsKeyRingsCryptoKeysService(self)
    self.projects_locations_keyRings = self.ProjectsLocationsKeyRingsService(self)
    self.projects_locations = self.ProjectsLocationsService(self)
    self.projects = self.ProjectsService(self)

  class ProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings_cryptoKeys_cryptoKeyVersions resource."""

    _NAME = u'projects_locations_keyRings_cryptoKeys_cryptoKeyVersions'

    def __init__(self, client):
      super(CloudkmsV1.ProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      """Create a new CryptoKeyVersion in a CryptoKey.

The server will assign the next sequential id. If unset,
state will be set to
ENABLED.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.create',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[],
        relative_path=u'v1/{+parent}/cryptoKeyVersions',
        request_field=u'cryptoKeyVersion',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest',
        response_type_name=u'CryptoKeyVersion',
        supports_download=False,
    )

    def Destroy(self, request, global_params=None):
      """Schedule a CryptoKeyVersion for destruction.

Upon calling this method, CryptoKeyVersion.state will be set to
DESTROY_SCHEDULED
and destroy_time will be set to a time 24
hours in the future, at which point the state
will be changed to
DESTROYED, and the key
material will be irrevocably destroyed.

Before the destroy_time is reached,
RestoreCryptoKeyVersion may be called to reverse the process.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsDestroyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
      config = self.GetMethodConfig('Destroy')
      return self._RunMethod(
          config, request, global_params=global_params)

    Destroy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}:destroy',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.destroy',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}:destroy',
        request_field=u'destroyCryptoKeyVersionRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsDestroyRequest',
        response_type_name=u'CryptoKeyVersion',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      """Returns metadata for a given CryptoKeyVersion.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.get',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest',
        response_type_name=u'CryptoKeyVersion',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Lists CryptoKeyVersions.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCryptoKeyVersionsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.list',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[u'pageSize', u'pageToken'],
        relative_path=u'v1/{+parent}/cryptoKeyVersions',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest',
        response_type_name=u'ListCryptoKeyVersionsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      """Update a CryptoKeyVersion's metadata.

state may be changed between
ENABLED and
DISABLED using this
method. See DestroyCryptoKeyVersion and RestoreCryptoKeyVersion to
move between other states.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}',
        http_method=u'PATCH',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.patch',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[u'updateMask'],
        relative_path=u'v1/{+name}',
        request_field=u'cryptoKeyVersion',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest',
        response_type_name=u'CryptoKeyVersion',
        supports_download=False,
    )

    def Restore(self, request, global_params=None):
      """Restore a CryptoKeyVersion in the.
DESTROY_SCHEDULED,
state.

Upon restoration of the CryptoKeyVersion, state
will be set to DISABLED,
and destroy_time will be cleared.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
      config = self.GetMethodConfig('Restore')
      return self._RunMethod(
          config, request, global_params=global_params)

    Restore.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}:restore',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.restore',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}:restore',
        request_field=u'restoreCryptoKeyVersionRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest',
        response_type_name=u'CryptoKeyVersion',
        supports_download=False,
    )

  class ProjectsLocationsKeyRingsCryptoKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings_cryptoKeys resource."""

    _NAME = u'projects_locations_keyRings_cryptoKeys'

    def __init__(self, client):
      super(CloudkmsV1.ProjectsLocationsKeyRingsCryptoKeysService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      """Create a new CryptoKey within a KeyRing.

CryptoKey.purpose is required.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.create',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[u'cryptoKeyId'],
        relative_path=u'v1/{+parent}/cryptoKeys',
        request_field=u'cryptoKey',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest',
        response_type_name=u'CryptoKey',
        supports_download=False,
    )

    def Decrypt(self, request, global_params=None):
      """Decrypts data that was protected by Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DecryptResponse) The response message.
      """
      config = self.GetMethodConfig('Decrypt')
      return self._RunMethod(
          config, request, global_params=global_params)

    Decrypt.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:decrypt',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.decrypt',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}:decrypt',
        request_field=u'decryptRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest',
        response_type_name=u'DecryptResponse',
        supports_download=False,
    )

    def Encrypt(self, request, global_params=None):
      """Encrypts data, so that it can only be recovered by a call to Decrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EncryptResponse) The response message.
      """
      config = self.GetMethodConfig('Encrypt')
      return self._RunMethod(
          config, request, global_params=global_params)

    Encrypt.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:encrypt',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.encrypt',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}:encrypt',
        request_field=u'encryptRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest',
        response_type_name=u'EncryptResponse',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      """Returns metadata for a given CryptoKey, as well as its.
primary CryptoKeyVersion.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.get',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest',
        response_type_name=u'CryptoKey',
        supports_download=False,
    )

    def GetIamPolicy(self, request, global_params=None):
      """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
      config = self.GetMethodConfig('GetIamPolicy')
      return self._RunMethod(
          config, request, global_params=global_params)

    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:getIamPolicy',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.getIamPolicy',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:getIamPolicy',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysGetIamPolicyRequest',
        response_type_name=u'Policy',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Lists CryptoKeys.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCryptoKeysResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.list',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[u'pageSize', u'pageToken'],
        relative_path=u'v1/{+parent}/cryptoKeys',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysListRequest',
        response_type_name=u'ListCryptoKeysResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      """Update a CryptoKey.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}',
        http_method=u'PATCH',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.patch',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[u'updateMask'],
        relative_path=u'v1/{+name}',
        request_field=u'cryptoKey',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest',
        response_type_name=u'CryptoKey',
        supports_download=False,
    )

    def SetIamPolicy(self, request, global_params=None):
      """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
      config = self.GetMethodConfig('SetIamPolicy')
      return self._RunMethod(
          config, request, global_params=global_params)

    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:setIamPolicy',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.setIamPolicy',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:setIamPolicy',
        request_field=u'setIamPolicyRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysSetIamPolicyRequest',
        response_type_name=u'Policy',
        supports_download=False,
    )

    def TestIamPermissions(self, request, global_params=None):
      """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
      config = self.GetMethodConfig('TestIamPermissions')
      return self._RunMethod(
          config, request, global_params=global_params)

    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:testIamPermissions',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.testIamPermissions',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:testIamPermissions',
        request_field=u'testIamPermissionsRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysTestIamPermissionsRequest',
        response_type_name=u'TestIamPermissionsResponse',
        supports_download=False,
    )

    def UpdatePrimaryVersion(self, request, global_params=None):
      """Update the version of a CryptoKey that will be used in Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
      config = self.GetMethodConfig('UpdatePrimaryVersion')
      return self._RunMethod(
          config, request, global_params=global_params)

    UpdatePrimaryVersion.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:updatePrimaryVersion',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.updatePrimaryVersion',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}:updatePrimaryVersion',
        request_field=u'updateCryptoKeyPrimaryVersionRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest',
        response_type_name=u'CryptoKey',
        supports_download=False,
    )

  class ProjectsLocationsKeyRingsService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings resource."""

    _NAME = u'projects_locations_keyRings'

    def __init__(self, client):
      super(CloudkmsV1.ProjectsLocationsKeyRingsService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      """Create a new KeyRing in a given Project and Location.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KeyRing) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.create',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[u'keyRingId'],
        relative_path=u'v1/{+parent}/keyRings',
        request_field=u'keyRing',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsCreateRequest',
        response_type_name=u'KeyRing',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      """Returns metadata for a given KeyRing.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KeyRing) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.get',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsGetRequest',
        response_type_name=u'KeyRing',
        supports_download=False,
    )

    def GetIamPolicy(self, request, global_params=None):
      """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
      config = self.GetMethodConfig('GetIamPolicy')
      return self._RunMethod(
          config, request, global_params=global_params)

    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:getIamPolicy',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.getIamPolicy',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:getIamPolicy',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest',
        response_type_name=u'Policy',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Lists KeyRings.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListKeyRingsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.keyRings.list',
        ordered_params=[u'parent'],
        path_params=[u'parent'],
        query_params=[u'pageSize', u'pageToken'],
        relative_path=u'v1/{+parent}/keyRings',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsListRequest',
        response_type_name=u'ListKeyRingsResponse',
        supports_download=False,
    )

    def SetIamPolicy(self, request, global_params=None):
      """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
      config = self.GetMethodConfig('SetIamPolicy')
      return self._RunMethod(
          config, request, global_params=global_params)

    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:setIamPolicy',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.setIamPolicy',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:setIamPolicy',
        request_field=u'setIamPolicyRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsSetIamPolicyRequest',
        response_type_name=u'Policy',
        supports_download=False,
    )

    def TestIamPermissions(self, request, global_params=None):
      """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
      config = self.GetMethodConfig('TestIamPermissions')
      return self._RunMethod(
          config, request, global_params=global_params)

    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:testIamPermissions',
        http_method=u'POST',
        method_id=u'cloudkms.projects.locations.keyRings.testIamPermissions',
        ordered_params=[u'resource'],
        path_params=[u'resource'],
        query_params=[],
        relative_path=u'v1/{+resource}:testIamPermissions',
        request_field=u'testIamPermissionsRequest',
        request_type_name=u'CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest',
        response_type_name=u'TestIamPermissionsResponse',
        supports_download=False,
    )

  class ProjectsLocationsService(base_api.BaseApiService):
    """Service class for the projects_locations resource."""

    _NAME = u'projects_locations'

    def __init__(self, client):
      super(CloudkmsV1.ProjectsLocationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      """Get information about a location.

      Args:
        request: (CloudkmsProjectsLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Location) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations/{locationsId}',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.get',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[],
        relative_path=u'v1/{+name}',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsGetRequest',
        response_type_name=u'Location',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Lists information about the supported locations for this service.

      Args:
        request: (CloudkmsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path=u'v1/projects/{projectsId}/locations',
        http_method=u'GET',
        method_id=u'cloudkms.projects.locations.list',
        ordered_params=[u'name'],
        path_params=[u'name'],
        query_params=[u'filter', u'pageSize', u'pageToken'],
        relative_path=u'v1/{+name}/locations',
        request_field='',
        request_type_name=u'CloudkmsProjectsLocationsListRequest',
        response_type_name=u'ListLocationsResponse',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = u'projects'

    def __init__(self, client):
      super(CloudkmsV1.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }
