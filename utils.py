


def get_model_size(model, trainable):
    if trainable:
        # torch. numel ( input ) → int: Returns the total number of elements in the input tensor.
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params

