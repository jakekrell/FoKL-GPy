"""
Routines for analyzing TCLab optimization experiments, with FoKL-GPy.

Developers:
- Jacob P. Krell (JPK), West Virginia University

| Developer | Date       | Changes                                                          |
|-----------|------------|------------------------------------------------------------------|
| JPK       | 2024-08-18 | first pass at porting Jupyter notebook methods into class object |

"""
import pandas as pd


class TCLab:
    def __init__(self, filenames):
        """
        | Argument  | Type          | Description                               |
        |-----------|---------------|-------------------------------------------|
        | filenames | list of str's | filenames of csv TCLab results to analyze |
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        if not isinstance(filenames, list):
            raise TypeError("'filenames' must be a list of strings.")
        for filename in filenames:
            if not isinstance(filename, str):
                raise TypeError("Each filename in 'filenmaes' must be a string.")
        
        # =======================================================================

        self.filenames = filenames

        self.data = {}
        for filename in self.filenames:
            self.data.update({filenames, pd.read_csv(filename)})
        
        return
    
    def func(self):
        return


