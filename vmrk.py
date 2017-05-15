#!/usr/bin/python

"""
Process one VMRK file into a set of summary statistics.

TODO optionally filter outliers
TODO allow selected blocks to be dropped

The processing is done in two steps.  First run process_vmrk to obtain
a condensed form of the VMRK data, then run summarize_vmrk to obtain
the summary statistics.
"""

import csv
import numpy as np
import os
import sys
import logging
from collections import OrderedDict

class Code(object):
    """
    A stimulus/response code
    """
    def __init__(self, side, congruent, correct):
        self.side = side
        self.congruent = congruent
        self.correct = correct

    def __str__(self):
        return "side=%s congruent=%r correct=%r" % (
            self.side, self.congruent, self.correct)

    def fromSRCodes(a, b):
        if a in (1, 2):
            side = "left"
        else:
            side = "right"
        if a in (1, 3):
            congruent = True
        else:
            congruent = False
        if b in (5, 6):
            correct = True
        else:
            correct = False

        return Code(side, congruent, correct)


class Trial(object):
    """
    Raw information about one trial.
    """
    def __init__(self, srcode, time):
        self.srcode = srcode
        self.time = time

    def __str__(self):
        return "srcode=%s time=%d" % (self.srcode, self.time)


class Block(object):
    """
    Holds all information about a block of trials
    """
    def __init__(self):
        self.Code = []    # Stimulus/response codes
        self.Rtim = []    # Response times
        self.Ntri = []    # Number of responses within a trial

    def query(self, side=None, congruent=None, correct=None, lastcorrect=None):
        """
        Return all response times with a given stimulus/response code pair.
        """
        ret = []
        for j, (c, x) in enumerate(zip(self.Code, self.Rtim)):
            if side is not None and c.side != side:
                continue
            if congruent is not None and c.congruent != congruent:
                continue
            if correct is not None and c.correct != correct:
                continue
            if lastcorrect is not None and j > 0:
                if self.Code[j-1].correct != lastcorrect:
                    continue
            ret.append(x)

        return ret

    def postCorrectRtim(self):
        """
        Return all response times that follow a correct response.
        """
        ret = []
        for k in range(1, len(self.Rtim)):
            if self.KY[k-1][1] in (5, 6):
                ret.append(self.Rtim[k])
        return ret

    def postErrorRtim(self):
        """
        Return all response times that follow an error response.
        """
        ret = []
        for k in range(1, len(self.Rtim)):
            if self.KY[k-1][1] in (5, 6):
                ret.append(self.Rtim[k])
        return ret

    def __str__(self):
        print(self.RT)

def process_trial(qu, block):
    """
    Insert summary values obtained from qu into block.

    Parameters
    ----------
    qu :
        Data from a single trial.
    block :
        All data grouped by trial/response type
    """

    # A trial must start with S99 (fixation cross), then have a
    # stimulus and response.  Return early if this is not the case.
    if len(qu) < 3 or qu[0].srcode != 99:
        return

    # ky is the stimulus and response type
    code = Code.fromSRCodes(qu[1].srcode, qu[2].srcode)

    # Response time for first response, multiplication by 2 is a scale
    # conversion.
    rt = 2 * (qu[2].time - qu[1].time)
    block.Code.append(code)
    block.Rtim.append(rt)
    block.Ntri.append(len(qu))


def collapse_blocks(data):
    """
    Collapse a list of blocks into a single block.
    """
    blk = Block()
    for block in data:
        blk.Code.extend(block.Code)
        blk.Rtim.extend(block.Rtim)
        blk.Ntri.extend(block.Ntri)
    return blk


def summarize_vmrk(filename, data):
    """
    Create summary statistics from a VMRK file.

    The VMRK file must be processed with process_vmrk before running
    this function.
    """

    cdata = collapse_blocks(data)

    results = OrderedDict()

    results["sid"] = filename.split(".")[0]

    all_trials = cdata.Rtim
    correct_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if k.correct]
    error_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if not k.correct]

    results["fcn"] = len(correct_trials)
    results["fen"] = len(error_trials)
    results["facc"] = 100 * results["fcn"] / len(all_trials)

    # All trial summaries
    results["frtm"] = np.mean(all_trials)
    results["frtsd"] = np.std(all_trials, ddof=1)

    # All correct trial summaries
    results["frtmc"] = np.mean(correct_trials)
    results["frtsdc"] = np.std(correct_trials, ddof=1)

    # All error trial summaries
    results["frtme"] = np.mean(error_trials)
    results["frtsde"] = np.std(error_trials, ddof=1)

    # Congruent correct trials
    v = cdata.query(correct=True, congruent=True)
    results["fccn"] = len(v)
    results["fcrtmc"] = np.mean(v)
    results["fcrtsdc"] = np.std(v, ddof=1)

    # Congruent error trials
    v = cdata.query(correct=False, congruent=True)
    results["fcen"] = len(v)
    results["fcrtme"] = np.mean(v)
    results["fcrtsde"] = np.std(v, ddof=1)

    # Congruent accuracy
    results["fcacc"] = 100 * results["fccn"] / (results["fccn"] + results["fcen"])

    # Incongruent correct trials
    v = cdata.query(correct=True, congruent=False)
    results["ficn"] = len(v)
    results["firtmc"] = np.mean(v)
    results["firtsdc"] = np.std(v, ddof=1)

    # Incongruent error trials
    v = cdata.query(correct=False, congruent=False)
    results["fien"] = len(v)
    results["firtme"] = np.mean(v)
    results["firtsde"] = np.std(v, ddof=1)

    # Incongruent accuracy
    results["fiacc"] = 100 * results["ficn"] / (results["ficn"] + results["fien"])

    # Post correct correct trials
    # (don't count first trial of each block)
    v = [b.query(correct=True, lastcorrect=True) for b in data]
    results["fpccn"] = sum([len(x) - 1 for x in v])

    # Post correct error trials
    v = cdata.query(correct=False, lastcorrect=True)
    results["fpcen"] = len(v)

    # Post error correct trials
    # (don't count first trial of each block)
    v = [b.query(correct=True, lastcorrect=False) for b in data]
    results["fpecn"] = sum([len(x) - 1 for x in v])
    results["fpecrtm"] = sum([sum(x[1:]) for x in v]) / results["fpecn"]

    # Post error error trials
    v = cdata.query(correct=False, lastcorrect=False)
    results["fpeen"] = len(v)
    results["fpeertm"] = np.mean(v)

    # Post correct accuracy
    results["fpaccpc"] = results["fpccn"] / (results["fpccn"] + results["fpcen"])

    # Post error accuracy
    results["fpaccpe"] = results["fpecn"] / (results["fpecn"] + results["fpeen"])

    # Post error slowing
    results["fpes"] = results["fpeertm"] - results["fpecrtm"]

    # Anticipatory responses
    results["fan"] = np.sum(np.asarray(all_trials) < 150)
    results["faen"] = np.sum(np.asarray(error_trials) < 150)

    # Trials with extra responses
    results["fscn"] = sum(np.asarray(cdata.Ntri) > 3)

    return results


def process_vmrk(filename):
    """
    Process the VMRK format file with name filename.

    Parameters
    ----------
    filename :
        Name of a vmrk format file.

    Returns
    -------
    data : list of Blocks
        data[j] contains all the data for block j.
    """

    fid = open(filename)
    rdr = csv.reader(fid)

    # Keep track of which block we are in
    blocknum = 0
    dblock = 0

    # Assume that we start in practice mode
    mode = "practice"
    qu = []
    data = []
    block = Block()

    for line in rdr:

        # Only process "mark" lines
        if len(line) == 0 or not line[0].startswith("Mk"):
            logging.info("Skipping row: %s" % line)
            continue

        # Lines have format Mk###=type, where type=comment, stimulus
        f0 = line[0].split("=")
        fl = f0[1].lower()
        if fl == "comment":
            # Switch to (or remain in) "experiment" mode
            if "experiment" in line[1].lower():
                mode = "experiment"
            continue
        elif fl == "stimulus":
            pass
        else:
            # Not sure what else exists, log it and move on
            logging.info("Skipping row: %s" % line)
            continue

        if mode != "experiment":
            continue

        # Get the type code, e.g. if S16 then n=16
        f1 = line[1].replace(" ", "")
        n = int(f1[1:])
        qu.append(Trial(n, int(line[2])))

        # Handle end of block markers
        if n in (144, 255):
            if dblock > 0:
                process_trial(qu[0:-1], block)
                qu = [qu[-1]]
                blocknum += 1
                dblock = 0
                data.append(block)
                block = Block()
            continue
        dblock += 1

        if n == 99:
            process_trial(qu[0:-1], block)
            qu = [qu[-1]]

    # Final trial may not have been processed
    process_trial(qu, block)

    return data


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("no files")
        sys.exit(0)

    import csv

    logging.basicConfig(filename="vmrk.log",
                        level=logging.DEBUG)

    results = []
    for i, fname in enumerate(sys.argv[1:]):

        data = process_vmrk(fname)
        result = summarize_vmrk(fname, data)

        if i == 0:
            wtr = csv.writer(sys.stdout)
            header = [k for k in result]
            wtr.writerow(header)

        wtr.writerow([result[k] for k in result])
