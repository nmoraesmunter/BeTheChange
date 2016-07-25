import requests
import exceptions as ex
import os

class ChangeOrgApi(object):
	"""
	This is the main Change.org API module, which encapsulates API methods
	"""


	def __init__(self, **kwagrs):
		"""
		Creates a new Api object. The API key and secret are passed through as arguments to this
		constructor, or can alternatively be set as environment variables, CHANGE_ORG_API_KEY
		and CHANGE_ORG_API_SECRET for the API key and secret respectively
		Args:
			kwagrs: {dict} Dictionary of arugments, with keys 'key' and 'secret', containing the
				change.org API key and secret
		Raises:
			ApiInitializationError: Thrown if API key and secret are not passed through correctly
		"""
		self.__BASEURL = 'https://api.change.org/v1/'
		api_key = self.__checkEnvironmentAndDict(kwagrs, 'api_key', 'CHANGE_ORG_API_KEY')
		api_secret = self.__checkEnvironmentAndDict(kwagrs, 'secret', 'CHANGE_ORG_API_SECRET')
		if api_key.get('exists') and api_secret.get('exists'):
			self.__KEY = api_key.get('value')
			self.__SECRET = api_secret.get('value')
		else:
			raise ex.ApiInitializationError

	def __checkEnvironmentAndDict(self, candidateDict, dictKey, osKey):
		"""
		Function to check if a specific key is present in a dictionary, and if a variable is
		present as an environment variable
		Args:
			candidateDict: {dict} Dictionary in which the key is to be searched for
			dictKey: {str} Key to be searched for in the dictionary
			osKey: {str} Key of desired environment variable
		Returns:
			{dict} A dict with keys 'exists' and 'value', where exists is a boolean (true if the value
			exists in either the dictionary or the environment variables or both, false if not) and
			where 'value' is the value retrieved from the dictionary or environment variables for the
			specified keys (if it exists, an empty string is returned if not). Note: if both
			environment and dictionary values are present, the value from the dictionary is returned
		"""
		if dictKey in candidateDict:
			return {'exists': True, 'value': candidateDict.get(dictKey)}
		else:
			candidateKey = os.environ.get(osKey)
			if candidateKey is not None:
				return {'exists': True, 'value': candidateKey}
			else:
				return {'exists': False, 'value': ''}

	def __makeRequest(self, url, params):
		"""
		Function to make a request to the change.org API, using the requests module.
		This function automatically adds the API key and secret to the request
		Args:
			url: {str} URL path to the requested resource (without base URL and first slash)
			params: {dict} Any additional request parameters apart from the API key and secret
		Returns:
			{dict} A dict of the response data from the API call. If there is an error, the error
			response from the server is returned
		Raises:
		"""
		request_url = self.__BASEURL + url
		params['api_key'] = self.__KEY
		try:
			r = requests.get(request_url, params=params)
			return r.json()
		except Exception:
			raise ex.ApiResponseError(self, str(Exception))

	def getPetitionId(self, petitionUrl):
		"""
		Function to get the ID of a petition, given the petition URL
		Args:
			petitionUrl: {str} URL of the petition
		Returns:
			{int} ID of the petition
		Raises:
			ApiResponseError: Thrown if there is an error in the API response
		"""
		resource_endpoint = 'petitions/get_id'
		params = {'petition_url': petitionUrl}
		res = self.__makeRequest(resource_endpoint, params)
		return int(res['petition_id'])

	def getSinglePetitionById(self, petitionId):
		"""
		Function to get details about a petition from its petition ID
		Args:
			petitionId: {int} ID of the petition
		Returns:
			{dict} A dict with details about the petition (title, status, url, overview
			targets, letter_body, signature_count, image_url, category, goal, created_at,
			end_at, creator_name, creator_url, organization_name, organization_url)
		Raises:
			ApiResponseError: Thrown if there is an error in the API response
			ApiRequestError: Thrown if there is an error with the request
							 (usually due to an incorrect petition ID)
		"""
		resource_endpoint = 'petitions/' + str(petitionId)
		res = self.__makeRequest(resource_endpoint, {})
		if 'result' in res.viewkeys():
			if res['result'] == 'failure':
				raise ex.ApiRequestError(self, res['error'])
		return res


	def getUserById(self, userId):
		"""
		Function to get details about a user from its user ID
		Args:
			userId: {int} ID of the user
		Returns:
			{dict} A dict with details about the user
		Raises:
			ApiResponseError: Thrown if there is an error in the API response
			ApiRequestError: Thrown if there is an error with the request
							 (usually due to an incorrect user ID)
		"""
		resource_endpoint = 'users/' + str(userId)
		res = self.__makeRequest(resource_endpoint, {})
		if 'result' in res.viewkeys():
			if res['result'] == 'failure':
				raise ex.ApiRequestError(self, res['error'])
		return res

	def getOrgById(self, orgId):
		"""
        Function to get details about an organization from its organization ID
        Args:
            orgId: {int} ID of the user
        Returns:
            {dict} A dict with details about the organization
        Raises:
            ApiResponseError: Thrown if there is an error in the API response
            ApiRequestError: Thrown if there is an error with the request
                             (usually due to an incorrect user ID)
        """
		resource_endpoint = 'organizations/' + str(orgId)
		res = self.__makeRequest(resource_endpoint, {})
		if 'result' in res.viewkeys():
			if res['result'] == 'failure':
				raise ex.ApiRequestError(self, res['error'])
		return res




	def getMultiplePetitionsById(self, petitionIds):
		"""
		Function to get details about multiple petitions from their petition IDs
		Args:
			petitionIds: {array} Array of petition IDs
		Returns:
			{array} Array of dict objects, with details about each of the petition
			IDs provoded, in sequential order
		Raises:
			ApiResponseError: Thrown if there is an error in the API response
			ApiRequestError: Thrown if there is an error in the API request
							 (usually due to an inccorect petition ID)
		"""
		resource_endpoint = 'petitions/'
		params = {'petition_ids': ''}
		for petition_id in petitionIds:
			params['petition_ids'] = params['petition_ids'] + str(petition_id) + ','
		res = self.__makeRequest(resource_endpoint, params)
		if 'result' in res.viewkeys():
			if res['result'] == 'failure':
				raise ex.ApiRequestError(self, res['error'])
		output = res['petitions']
		if res['total_pages'] > 1:
			next_page_endpoint = res['next_page_endpoint']
			while next_page_endpoint is not None:
				current_page = self.__makeRequest(next_page_endpoint, {})
				output = output + current_page['petitions']
				next_page_endpoint = current_page['next_page_endpoint']
		return output

	def getSignatureCountOnPetition(self, petitionId):
		"""
		Function to get the number of signatures on a petition
		Args:
			petitionId: {int} ID of the petition
		Returns:
			{int} number of signatures on the petition
		Raises:
			ApiResponseError: Thrown if there is an error in the API response
			ApiRequestError: Thrown if there is an error in the API request
							 (usually due to an incorrect petition ID)
		"""
		resource_endpoint = 'petitions/' + str(petitionId) + '/signatures'
		res = self.__makeRequest(resource_endpoint, {'page_size': 500})
		if 'result' in res.viewkeys():
			if res['result'] == 'failure':
				raise ex.ApiRequestError(self, res['error'])
		return res['signature_count']
