<?php

namespace App\Livewire;

use App\Models\User;
use App\Services\PaymentService;
use Livewire\Component;

class WalletScreen extends Component
{
    public $balance_btc = 1;

    public $payment_request;

    public $received_payments;

    public $lightning_address;

    public $payins; // Add this line

    protected $rules = [
        'payment_request' => 'required|string',
    ];

    // On mount, grab the user's bitcoin balance and payins
    public function mount()
    {
        // If the user is not logged in, redirect to the homepage
        if (! auth()->check()) {
            return redirect()->route('home');
        }

        /** @var User $user */
        $user = auth()->user();
        $this->balance_btc = $user->getSatsBalanceAttribute();
        $this->received_payments = $user->receivedPayments()->get()->reverse();

        // $this->lightning_address is the user's username
        $prefix = $user->username;
        // staging.openagents.com if env is staging, otherwise openagents.com
        $suffix = env('APP_ENV') === 'staging' ? 'staging.openagents.com' : 'openagents.com';
        $this->lightning_address = "{$prefix}@{$suffix}";

        // Fetch the payins
        $this->payins = $user->payins()->get()->reverse(); // Assumes there's a payins() relationship
    }

    public function submitPaymentRequest(PaymentService $paymentService): void
    {
        $this->validate();

        $response = $paymentService->processPaymentRequest($this->payment_request);

        if ($response['ok']) {
            session()->flash('message', 'Payment processed successfully.');
        } else {
            session()->flash('error', $response['error'] ?? 'Something went wrong.');
        }

        // Optionally update the balance and payins after processing the payment
        $user = auth()->user()->fresh();
        $this->balance_btc = $user->getSatsBalanceAttribute();
        $this->payins = $user->payins()->get()->reverse();
    }

    public function render()
    {
        return view('livewire.wallet-screen');
    }
}
