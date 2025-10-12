def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--classes', type=str, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='models/baseline_cnn.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & loaders
    dataset = SpeechCommandsSubset(args.data_root, args.classes)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)
    n_test  = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model & optimizer
    model = KWSCNN(num_classes=len(args.classes)).to(device)
    optimiz = optim.Adam(model.parameters(), lr=args.lr)

    # Create output dirs
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- stats containers for plotting ---
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        # One training epoch
        tl, ta = train_one_epoch(model, train_loader, optimiz, device)
        # Validation
        vl, va = evaluate(model, val_loader, device)

        # Record stats
        train_loss_list.append(tl); train_acc_list.append(ta)
        val_loss_list.append(vl);   val_acc_list.append(va)

        # Logging
        print(f"Epoch {epoch:02d}: train loss {tl:.4f} acc {ta*100:.2f}% | "
              f"val loss {vl:.4f} acc {va*100:.2f}%")

        # Save best model
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

    # --- plot curves after training finishes ---
    import matplotlib.pyplot as plt

    # Accuracy curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list,   label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig('results/train_val_accuracy_curve.png'); plt.close()

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list,   label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig('results/train_val_loss_curve.png'); plt.close()

    print("Training done. Best val acc: {:.2f}%".format(best_val * 100.0))
