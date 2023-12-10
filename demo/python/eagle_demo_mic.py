import argparse
import contextlib
import os
import struct
import time
import wave

import pveagle
from pvrecorder import PvRecorder

PV_RECORDER_FRAME_LENGTH = 512

FEEDBACK_TO_DESCRIPTIVE_MSG = {
    pveagle.EagleProfilerEnrollFeedback.AUDIO_OK: 'Good audio',
    pveagle.EagleProfilerEnrollFeedback.AUDIO_TOO_SHORT: 'Insufficient audio length',
    pveagle.EagleProfilerEnrollFeedback.UNKNOWN_SPEAKER: 'Different speaker in audio',
    pveagle.EagleProfilerEnrollFeedback.NO_VOICE_FOUND: 'No voice found in audio',
    pveagle.EagleProfilerEnrollFeedback.QUALITY_ISSUE: 'Low audio quality due to a bad microphone or environment'
}


class EnrollmentAnimation:
    def __init__(self, sleep_time_sec=0.1):
        self._sleep_time_sec = sleep_time_sec
        self._frames = [
            " .  ",
            " .. ",
            " ...",
            "  ..",
            "   .",
            "    "
        ]
        self._done = False
        self._percentage = 0
        self._feedback = ''

    def start(self):
        self._done = False
        while not self._done:
            for frame in self._frames:
                if self._done:
                    break
                print('\033[2K\033[1G\r[%3d%%]' % self._percentage + self._feedback + frame, end='', flush=True)
                time.sleep(self._sleep_time_sec)

    def stop(self):
        print('\033[2K\033[1G\r[%3d%%]' % self._percentage + self._feedback, end='', flush=True)
        self._done = True

    @property
    def percentage(self):
        return self._percentage

    @property
    def feedback(self):
        return self._feedback

    @percentage.setter
    def percentage(self, value):
        self._percentage = value

    @feedback.setter
    def feedback(self, value):
        self._feedback = value


def print_result(scores, labels):
    result = '\rscores -> '
    result += ', '.join('`%s`: %.2f' % (label, score) for label, score in zip(labels, scores))
    print(result, end='', flush=True)


def test_operation(args):
    profiles = list()
    speaker_labels = list()
    for profile_path in args.input_profile_paths:
        speaker_labels.append(os.path.splitext(os.path.basename(profile_path))[0])
        with open(profile_path, 'rb') as f:
            profile = pveagle.EagleProfile.from_bytes(f.read())
        profiles.append(profile)

    eagle = pveagle.create_recognizer(
        access_key=args.access_key,
        model_path=args.model_path,
        library_path=args.library_path,
        speaker_profiles=profiles)

    recorder = PvRecorder(device_index=args.audio_device_index, frame_length=eagle.frame_length)
    recorder.start()

    with contextlib.ExitStack() as file_stack:
        if args.output_audio_path is not None:
            test_audio_file = file_stack.enter_context(wave.open(args.output_audio_path, 'wb'))
            test_audio_file.setnchannels(1)
            test_audio_file.setsampwidth(2)
            test_audio_file.setframerate(eagle.sample_rate)

        print('Listening for audio... (press Ctrl+C to stop)')
        start_times = {label: None for label in speaker_labels}
        enrollment_animation = EnrollmentAnimation()

        try:
            recording_active = {label: False for label in speaker_labels}
            start_times = {label: None for label in speaker_labels}
            speech_buffer = {label: [] for label in speaker_labels}

            while True:
                pcm = recorder.read()
                if args.output_audio_path is not None:
                    for label, active in recording_active.items():
                        if active:
                            test_audio_file.writeframes(struct.pack('%dh' % len(pcm), *pcm))

                scores = eagle.process(pcm)
                print_result(scores, speaker_labels)

                for label, confidence in zip(speaker_labels, scores):
                    if confidence > 0.0:
                        if not recording_active[label]:
                            recording_active[label] = True
                            start_times[label] = time.time()
                            #start of speech, initialize audio buffer
                            speech_buffer[label] = []
                        else:
                            #append audio to buffer during speech
                            speech_buffer[label].extend(pcm)
                    elif recording_active[label]:
                        #check for end of speech
                        end_time = time.time()
                        duration = end_time - start_times[label]
                        if duration > args.min_speech_duration:
                            #here i save the audio buffer associated with the speech
                            export_path = f"{label}_speech_{int(start_times[label])}.wav"
                            with wave.open(export_path, 'wb') as export_file:
                                export_file.setnchannels(1)
                                export_file.setsampwidth(2)
                                export_file.setframerate(eagle.sample_rate)
                                export_file.writeframes(struct.pack('%dh' % len(speech_buffer[label]), *speech_buffer[label]))
                            print(f"\nSpeaker '{label}' talked with confidence > 0.0 for {duration:.2f} seconds\n"
                                  f"Audio saved to: {export_path}\n")
                        recording_active[label] = False

        except KeyboardInterrupt:
            print('\nStopping...')
        except pveagle.EagleActivationLimitError:
            print('\nAccessKey has reached its processing limit')
        finally:
            if eagle is not None:
                eagle.delete()
            recorder.stop()
            recorder.delete()




def main():
    parser = argparse.ArgumentParser()
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--access_key',
        required=True,
        help='AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)')
    common_parser.add_argument(
        '--library_path',
        help='Absolute path to a dynamic library. Default: using the library provided by `pveagle`')
    common_parser.add_argument(
        '--model_path',
        help='Absolute path to Eagle model. Default: using the model provided by `pveagle`')
    common_parser.add_argument('--audio_device_index', type=int, default=-1, help='Index of input audio device')
    common_parser.add_argument(
        '--output_audio_path',
        help='If provided, all recorded audio data will be saved to the given .wav file')
    common_parser.add_argument(
        '--min_speech_duration',
        type=float,
        default=4.0,
        help='Minimum duration (in seconds) for speech segments to be saved. Default: 2.0')

    subparsers = parser.add_subparsers(dest='command')

    test = subparsers.add_parser(
        'test',
        help='Evaluate Eagle''s performance using the provided speaker profiles.',
        parents=[common_parser])
    test.add_argument(
        '--input_profile_paths',
        required=True,
        nargs='+',
        help='Absolute path(s) to speaker profile(s)')

    args = argparse.Namespace()  # Create an empty namespace to store the arguments
    args.show_audio_devices = False  # Add a default value to the new argument

    args = parser.parse_args(namespace=args)

    if args.show_audio_devices:
        for index, name in enumerate(PvRecorder.get_available_devices()):
            print('Device #%d: %s' % (index, name))
        return

    if args.command == 'test':
        test_operation(args)

    else:
        print('Please specify a mode: enroll or test')
        return


if __name__ == '__main__':
    main()
