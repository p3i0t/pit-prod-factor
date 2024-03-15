class CalendarMissingError(Exception):
    """Exception raised when a required calendar is missing."""
    def __init__(self, message="Calendar is missing"):
        self.message = message
        super().__init__(self.message)