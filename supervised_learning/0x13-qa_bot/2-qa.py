#!/usr/bin/env python3
"""Function that answers questions from a reference text"""

import cmd
quenstion_answer = __import__('0-qa').quenstion_answer


bye = ['exit', 'quit', 'goodbye', 'bye']


def answer_loop(reference):
    """Function that answers questions from a reference text"""
    reference = reference

    class QABotCommand(cmd.Cmd):
        """QAbotCommand class"""
        prompt = "Q: "

        def precmd(self, line):
            """This method is called after the line has been input but before
                it has been interpreted. If you want to modify the input line
                before execution (for example, variable substitution)
                do it here."""
            if line.lower() in bye:
                print("A: Goodbye")
                self.do_bye(line)
                return ("bye")
            else:
                answer = question_answer(line, reference)
                if answer is None:
                    answer = "Sorry, I do not understand your question."
                print("A:", answer)
                return " "

        def do_bye(self, arg):
            """Method to exit and say goodbye"""
            return True

        def emptyline(self):
            """Called when an empty line is entered in response to the prompt.
            If this method is not overridden, it repeats the last nonempty
            command entered.
            """
            pass

    QABotCommand().cmdloop()
