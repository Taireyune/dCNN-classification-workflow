from training.model_trainer import model_trainer

if __name__ == "__main__":
    name = "CatDog-ColorGenData-SeInRes"
    
    load_file = 'CatDog-ColorGenData-SeInRes-2e-4-0030.hdf5'
    
    train = model_trainer(name)
    train.setup_model(learning_rate=0.0001)
    loss, acc = train.reload_model(load_file)
    train.run_model(270, period = 10)
    #train.save_model(load_file)