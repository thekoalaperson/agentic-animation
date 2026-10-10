"""Check the selected Stable Audio model config without exposing credentials."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true',
                        help='Check official access and allow downloading the small model config')
    args = parser.parse_args()
    from huggingface_hub import get_token, hf_hub_download
    token = get_token()
    repository = 'stabilityai/stable-audio-3-small-sfx'
    try:
        path = hf_hub_download(repository, 'model_config.json', token=token,
                               local_files_only=not args.online)
        result = {'repo': repository, 'file': 'model_config.json',
                  'available': True, 'cachePath': path}
    except Exception as exc:
        response = getattr(exc, 'response', None)
        result = {'repo': repository, 'available': False,
                  'errorType': type(exc).__name__,
                  'httpStatus': getattr(response, 'status_code', None)}
    print(json.dumps({'existingCredentialPresent': bool(token),
                      'mode': 'online-access' if args.online else 'local-cache-only',
                      'checks': [result]}))


if __name__ == '__main__':
    main()
