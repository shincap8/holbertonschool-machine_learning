#!/usr/bin/env python3
"""Script that takes in input from the user with
the prompt Q: and prints A: as a response."""

import cmd


bye = ['exit', 'quit', 'goodbye', 'bye']


class QABotCommand(cmd.Cmd):
    """QAbotCommand class"""
    prompt = "Q: "

    def precmd(self, line):
        """This method is called after the line has been input but before
            it has been interpreted. If you want to modify the input line
            before execution (for example, variable substitution) do it here.
        """
        if line.lower() in bye:
            print("A: Goodbye")
            self.do_bye(line)
            return ("bye")
        else:
            print("A:")
            return " "

    def do_bye(self, arg):
        """Method to exit and say goodbye"""
        return True

    def do_EOF(self, line):
        """EOF command to exit the program"""
        return True

    def emptyline(self):
        """Called when an empty line is entered in response to the prompt.
        If this method is not overridden, it repeats the last nonempty
        command entered.
        """
        pass


if __name__ == "__main__":
    QABotCommand().cmdloop()
