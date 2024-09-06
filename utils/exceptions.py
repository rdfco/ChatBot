class CouldNotFindAnswerException(Exception):
    def __str__(self) -> str:
        return "Sorry, I could not find the answer."


class InternalServerErrorException(Exception):
    def __str__(self) -> str:
        return "Sorry, I encountered an internal server error."
