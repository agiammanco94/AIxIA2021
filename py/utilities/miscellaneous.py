# -*- coding: utf-8 -*-
"""
    This module contains various utilities function, mainly related to the navigation in the project folder structure,
    and to the management of time.
"""
import datetime
import math
import os
import subprocess
import time
from typing import List


class Timer:
    """
        This class models a simple timer, which can be used to monitor the execution time of code fragments.

        Attributes:
            start_time: A float representing the number of seconds elapsed from "epoch", i.e.,
            January 1, 1970, 00:00:00, when the timer is started.

            end_time: A float representing the number of seconds elapsed from "epoch", when the timer is ended.
            seconds_elapsed: A float representing the amount of seconds elapsed between the call to start and
            stop methods.

            minutes_elapsed: A float representing the amount of minutes elapsed between the call to start and
            stop methods.
    """
    def __init__(self) -> None:
        """
            Inits a Timer instance with 0 values for the attributes, and then starts it.
        """
        self.start_time = 0
        self.end_time = 0
        self.seconds_elapsed = 0
        self.minutes_elapsed = 0
        self.start()

    def start(self) -> None:
        """
            Starts the Timer recording the current time.
        """
        self.start_time = time.time()

    def stop(self) -> float:
        """
            Stops the Timer, returning the amount of minutes elapsed.

            Returns:
                minutes_elapsed: A float representing the amount of minutes elapsed between the start and stop of the
                    timer.
        """
        self.end_time = time.time()
        self.seconds_elapsed = (self.end_time - self.start_time)
        self.minutes_elapsed = self.seconds_elapsed / 60
        return self.minutes_elapsed


def print_with_timestamp(message: str) -> None:
    """
        Utility function to print a string with the current date and time.

        Args:
            message: A string representing the message to output with the current timestamp.
    """
    ts = datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')
    print(f'({ts}) - ', end='')
    print(message)


def create_dir(dir_path: str) -> None:
    """
        Utility function which creates a directory if it is not already present in the system.

        Args:
            dir_path: A string representing the path of the directory to create.
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def is_square(i: int) -> bool:
    """
        Utility function which returns True whether the input number is a perfect square.

        Args:
            i: An integer to check for it being a perfect square.

        Returns:
            square: A boolean indicating whether the input number is a perfect square.
    """
    square = i == math.isqrt(i) ** 2
    return square


def get_relative_path() -> str:
    """
        Returns the prefix to come back to the root folder.

        Returns:
            prefix: A string representing the path to the root folder of the project.
    """

    s = str(subprocess.check_output(['pwd']))
    s = s[:-2].split("/")[1:]
    s[-1] = s[-1][:-1]

    prefix = "/"

    while s[-1] != 'AIxIA2021':
        prefix += '../'
        s = s[:-1]

    current_path = os.getcwd()

    return current_path + prefix


def get_list_of_paths(dir_path: str) -> List[str]:
    """
        Given a certain directory, this function returns a list of all the sub-directories contained inside it.

        Args:
            dir_path: A string representing the path of the directory to scan.

        Returns:
            list_of_paths: A list of strings containing the paths of the sub-directories inside dir_path.
    """
    list_of_paths = [dir_path + '/' + x for x in os.listdir(dir_path) if os.path.isdir(dir_path + '/' + x)]
    return list_of_paths
