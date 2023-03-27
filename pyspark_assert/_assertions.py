class TemplatedAssertionError(AssertionError):
    """Assertion with custom messages based on templates."""

    msg_template: str

    def __init__(self, *args, **kwargs):
        msg = self.msg_template.format(*args, **kwargs)
        super().__init__(msg)


class IncorrectTypeAssertionError(TemplatedAssertionError):

    msg_template = '{0} is not a {1} ({2})'


class DifferentLengthAssertionError(TemplatedAssertionError):

    msg_template = (
        'Lengths are different\n'
        '\tExpected : {1}\n'
        '\tActual   : {0}'
    )


class DifferentSchemaAssertionError(TemplatedAssertionError):

    msg_template = (
        'Schemas are different\n'
        '\tExpected : {1}\n'
        '\tActual   : {0}'
    )


class DifferentDataAssertionError(TemplatedAssertionError):

    msg_template = (
        'Schemas are different\n'
        '\tExpected : {1}\n'
        '\tActual   : {0}'
    )

    def __init__(self, left, right):

        left_msg = self._construct_msg(left)
        right_msg = self._construct_msg(right)

        super().__init__(left_msg, right_msg)

    @staticmethod
    def _construct_msg(data):
        spacing = '\t\n           '
        if isinstance(data, list):
            return spacing + spacing.join(repr(x) for x in data) + '\n'
        if isinstance(data, dict):
            # Assuming it's a counter
            return spacing + spacing.join(f'{k} [x{n}]' for k, n in data.items()) + '\n'

        return repr(data)
