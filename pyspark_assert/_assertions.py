class TemplatedAssertionError(AssertionError):

    msg_template: str

    def __init__(self, *args, **kwargs):
        msg = self.msg_template.format(*args, **kwargs)
        super().__init__(msg)


class IncorrectTypeAssertionError(TemplatedAssertionError):

    msg_template = '{0} is not a DataFrame ({1})'


class DifferentLengthAssertionError(TemplatedAssertionError):

    msg_template = (
        'Lengths are different\n'
        '\tExpected :{1}\n'
        '\tActual   :{0}'
    )


class DifferentSchemaAssertionError(TemplatedAssertionError):

    msg_template = (
        'Schemas are different\n'
        '\tExpected :{1}\n'
        '\tActual   :{0}'
    )
