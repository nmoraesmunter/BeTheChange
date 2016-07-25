class ApiValidationError(Exception):
	"""
	Exception class for an error to be thrown fi the API key and/or secret cannot be validated
	with the REST API
	"""
	def __str__(self):
		return 'Could not verify API key and/or API secret'

class ApiInitializationError(Exception):
	"""
	Exception class for an error to be thrown if the API module cannot be initialized
	"""
	def __str__(self):
		return 'Could not initailize change.org API module. Check API key and secret are properly passed through'

class ApiResponseError(Exception):
	"""
	Exception class for an error to be thrown if the response of the REST API is not what is
	expected. This is usually caused by a bad request
	"""
	def __init__(self, expr, msg):
		self.msg = msg
	def __str__(self):
		return 'Unexpected response from REST API. Response message below.\n\n' + self.msg

class ApiRequestError(Exception):
	"""
	Exception class for an error to be thrown if there is an error when making a request
	to the REST API
	"""
	def __init__(self, expr, msg):
		self.msg = msg
	def __str__(self):
		return 'Error in making request to the server. See exception message below.\n\n' + self.msg