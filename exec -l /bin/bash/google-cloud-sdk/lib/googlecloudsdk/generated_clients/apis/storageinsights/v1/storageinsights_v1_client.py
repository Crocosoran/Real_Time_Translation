"""Generated client library for storageinsights version v1."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storageinsights.v1 import storageinsights_v1_messages as messages


class StorageinsightsV1(base_api.BaseApiClient):
  """Generated client library for service storageinsights version v1."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://storageinsights.googleapis.com/'
  MTLS_BASE_URL = 'https://storageinsights.mtls.googleapis.com/'

  _PACKAGE = 'storageinsights'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v1'
  _CLIENT_ID = 'CLIENT_ID'
  _CLIENT_SECRET = 'CLIENT_SECRET'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'StorageinsightsV1'
  _URL_VERSION = 'v1'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new storageinsights handle."""
    url = url or self.BASE_URL
    super(StorageinsightsV1, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.projects_locations_datasetConfigs = self.ProjectsLocationsDatasetConfigsService(self)
    self.projects_locations_operations = self.ProjectsLocationsOperationsService(self)
    self.projects_locations_reportConfigs_reportDetails = self.ProjectsLocationsReportConfigsReportDetailsService(self)
    self.projects_locations_reportConfigs = self.ProjectsLocationsReportConfigsService(self)
    self.projects_locations = self.ProjectsLocationsService(self)
    self.projects = self.ProjectsService(self)

  class ProjectsLocationsDatasetConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasetConfigs resource."""

    _NAME = 'projects_locations_datasetConfigs'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsLocationsDatasetConfigsService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      r"""Create method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs',
        http_method='POST',
        method_id='storageinsights.projects.locations.datasetConfigs.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['datasetConfigId', 'requestId'],
        relative_path='v1/{+parent}/datasetConfigs',
        request_field='datasetConfig',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsCreateRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Delete method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}',
        http_method='DELETE',
        method_id='storageinsights.projects.locations.datasetConfigs.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['requestId'],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsDeleteRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Get method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatasetConfig) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}',
        http_method='GET',
        method_id='storageinsights.projects.locations.datasetConfigs.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsGetRequest',
        response_type_name='DatasetConfig',
        supports_download=False,
    )

    def LinkDataset(self, request, global_params=None):
      r"""LinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('LinkDataset')
      return self._RunMethod(
          config, request, global_params=global_params)

    LinkDataset.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}:linkDataset',
        http_method='POST',
        method_id='storageinsights.projects.locations.datasetConfigs.linkDataset',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:linkDataset',
        request_field='linkDatasetRequest',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""List method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatasetConfigsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs',
        http_method='GET',
        method_id='storageinsights.projects.locations.datasetConfigs.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['filter', 'orderBy', 'pageSize', 'pageToken'],
        relative_path='v1/{+parent}/datasetConfigs',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsListRequest',
        response_type_name='ListDatasetConfigsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Patch method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}',
        http_method='PATCH',
        method_id='storageinsights.projects.locations.datasetConfigs.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['requestId', 'updateMask'],
        relative_path='v1/{+name}',
        request_field='datasetConfig',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsPatchRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def UnlinkDataset(self, request, global_params=None):
      r"""UnlinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('UnlinkDataset')
      return self._RunMethod(
          config, request, global_params=global_params)

    UnlinkDataset.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}:unlinkDataset',
        http_method='POST',
        method_id='storageinsights.projects.locations.datasetConfigs.unlinkDataset',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:unlinkDataset',
        request_field='unlinkDatasetRequest',
        request_type_name='StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class ProjectsLocationsOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_operations resource."""

    _NAME = 'projects_locations_operations'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsLocationsOperationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Cancel(self, request, global_params=None):
      r"""Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (StorageinsightsProjectsLocationsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
      config = self.GetMethodConfig('Cancel')
      return self._RunMethod(
          config, request, global_params=global_params)

    Cancel.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}:cancel',
        http_method='POST',
        method_id='storageinsights.projects.locations.operations.cancel',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:cancel',
        request_field='cancelOperationRequest',
        request_type_name='StorageinsightsProjectsLocationsOperationsCancelRequest',
        response_type_name='Empty',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (StorageinsightsProjectsLocationsOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}',
        http_method='DELETE',
        method_id='storageinsights.projects.locations.operations.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsOperationsDeleteRequest',
        response_type_name='Empty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (StorageinsightsProjectsLocationsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}',
        http_method='GET',
        method_id='storageinsights.projects.locations.operations.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsOperationsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (StorageinsightsProjectsLocationsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations',
        http_method='GET',
        method_id='storageinsights.projects.locations.operations.list',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['filter', 'pageSize', 'pageToken'],
        relative_path='v1/{+name}/operations',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsOperationsListRequest',
        response_type_name='ListOperationsResponse',
        supports_download=False,
    )

  class ProjectsLocationsReportConfigsReportDetailsService(base_api.BaseApiService):
    """Service class for the projects_locations_reportConfigs_reportDetails resource."""

    _NAME = 'projects_locations_reportConfigs_reportDetails'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsLocationsReportConfigsReportDetailsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Gets details of a single ReportDetail.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsReportDetailsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportDetail) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}/reportDetails/{reportDetailsId}',
        http_method='GET',
        method_id='storageinsights.projects.locations.reportConfigs.reportDetails.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsReportDetailsGetRequest',
        response_type_name='ReportDetail',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists ReportDetails in a given project and location.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsReportDetailsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReportDetailsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}/reportDetails',
        http_method='GET',
        method_id='storageinsights.projects.locations.reportConfigs.reportDetails.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['filter', 'orderBy', 'pageSize', 'pageToken'],
        relative_path='v1/{+parent}/reportDetails',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsReportDetailsListRequest',
        response_type_name='ListReportDetailsResponse',
        supports_download=False,
    )

  class ProjectsLocationsReportConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_reportConfigs resource."""

    _NAME = 'projects_locations_reportConfigs'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsLocationsReportConfigsService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      r"""Creates a new ReportConfig in a given project and location.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs',
        http_method='POST',
        method_id='storageinsights.projects.locations.reportConfigs.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['requestId'],
        relative_path='v1/{+parent}/reportConfigs',
        request_field='reportConfig',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsCreateRequest',
        response_type_name='ReportConfig',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}',
        http_method='DELETE',
        method_id='storageinsights.projects.locations.reportConfigs.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['force', 'requestId'],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsDeleteRequest',
        response_type_name='Empty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets details of a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}',
        http_method='GET',
        method_id='storageinsights.projects.locations.reportConfigs.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsGetRequest',
        response_type_name='ReportConfig',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists ReportConfigs in a given project and location.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReportConfigsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs',
        http_method='GET',
        method_id='storageinsights.projects.locations.reportConfigs.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['filter', 'orderBy', 'pageSize', 'pageToken'],
        relative_path='v1/{+parent}/reportConfigs',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsListRequest',
        response_type_name='ListReportConfigsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates the parameters of a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}',
        http_method='PATCH',
        method_id='storageinsights.projects.locations.reportConfigs.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['requestId', 'updateMask'],
        relative_path='v1/{+name}',
        request_field='reportConfig',
        request_type_name='StorageinsightsProjectsLocationsReportConfigsPatchRequest',
        response_type_name='ReportConfig',
        supports_download=False,
    )

  class ProjectsLocationsService(base_api.BaseApiService):
    """Service class for the projects_locations resource."""

    _NAME = 'projects_locations'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsLocationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Gets information about a location.

      Args:
        request: (StorageinsightsProjectsLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Location) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations/{locationsId}',
        http_method='GET',
        method_id='storageinsights.projects.locations.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsGetRequest',
        response_type_name='Location',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists information about the supported locations for this service.

      Args:
        request: (StorageinsightsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/locations',
        http_method='GET',
        method_id='storageinsights.projects.locations.list',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['filter', 'pageSize', 'pageToken'],
        relative_path='v1/{+name}/locations',
        request_field='',
        request_type_name='StorageinsightsProjectsLocationsListRequest',
        response_type_name='ListLocationsResponse',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = 'projects'

    def __init__(self, client):
      super(StorageinsightsV1.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }
