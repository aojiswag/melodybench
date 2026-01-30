import torchcrepe
import torch
import resampy

def pitchpred(src, cuda):
    
    y = resampy.resample(y, sr, 16000)
    sr = 16000

    audio = torch.from_numpy(y).float().unsqueeze(0)

    hop_length = 80

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 2000

    # Select a model capacity--one of "tiny" or "full"
    model = 'full'

    if cuda:
        device = 'cuda:0'
        batch_size = 2048
        print("crepe run as cuda")
    else:
        device = 'cpu'
        batch_size = 512
        print("crepe run as cpu")

    # Compute pitch using first gpu
    pitch, confidence= torchcrepe.predict(audio,
                            sr,
                            hop_length,
                            fmin,
                            fmax,
                            model,
                            batch_size=batch_size,
                            device=device,
                            return_periodicity=True
                            )

    