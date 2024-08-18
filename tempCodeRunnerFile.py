    # Kalanları hesaplama
        train_errors = y_train - y_train_pred
        test_errors = y_test - y_test_pred

        # Subplot oluşturma
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Regresyon Modelinin Analizi', fontsize=16)

        # Eğitim verisi grafiği
        axs[0, 0].scatter(X_train.flatten(), y_train, color='blue', label='Gerçek Eğitim Verileri')
        axs[0, 0].plot(X_train.flatten(), y_train_pred, 'r-', label='Tahmin Edilen Eğitim Verileri')
        axs[0, 0].set_title('Eğitim Verileri: Öngörü vs. Gerçek Değerler')
        axs[0, 0].legend()

        # Test verisi grafiği
        axs[0, 1].scatter(X_test.flatten(), y_test, color='blue', label='Gerçek Test Verileri')
        axs[0, 1].plot(X_test.flatten(), y_test_pred, 'r-', label='Tahmin Edilen Test Verileri')
        axs[0, 1].set_title('Test Verileri: Öngörü vs. Gerçek Değerler')
        axs[0, 1].legend()

        # Kalan grafiği
        axs[1, 0].scatter(y_test_pred, test_errors, c='blue', label='Test Kalanları')
        axs[1, 0].axhline(0, color='red', linestyle='--')
        axs[1, 0].set_title('Test Kalanları vs. Tahmin Edilen Test Değerleri')
        axs[1, 0].set_xlabel('Tahmin Edilen Test Değerleri')
        axs[1, 0].set_ylabel('Kalanlar')

        # Hata dağılımı grafiği
        sns.histplot(test_errors, kde=True, ax=axs[1, 1])
        axs[1, 1].set_title('Test Hata Dağılımı')

        # Özniteliklerin etkisi grafiği (model katsayıları)
        coefficients = [model.intercept_, model.coef_[0]]
        features = ['Intercept', 'Öznitelik']
        axs[2, 0].bar(features, coefficients, color=['blue', 'green'])
        axs[2, 0].set_title('Özelliklerin Etkisi')

        # Öngörü aralıkları grafiği (test seti için)
        prediction_interval = 1.96 * np.std(test_errors)  # Örnek aralıklar
        axs[2, 0].plot(X_test.flatten(), y_test_pred.flatten(), 'r-', label='Tahmin Edilen Test Verileri')
        axs[2, 0].fill_between(X_test.flatten(), 
                            y_test_pred.flatten() - prediction_interval, 
                            y_test_pred.flatten() + prediction_interval, 
                            color='gray', alpha=0.2)
        axs[2, 0].scatter(X_test.flatten(), y_test, color='blue', label='Gerçek Test Verileri')
        axs[2, 0].set_title('Öngörü Aralıkları (Test Seti)')
        axs[2, 0].legend()

        # Boş subplot
        axs[2, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Üst başlık için boşluk bırakır
        plt.show()