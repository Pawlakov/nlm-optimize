namespace NLMBaseGUI.Behaviors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Avalonia;
    using Avalonia.Controls;
    using Avalonia.Data;
    using Avalonia.Interactivity;
    using Avalonia.Xaml.Interactivity;

    public class LostFocusUpdateBindingBehavior 
        : Behavior<TextBox>
    {
        static LostFocusUpdateBindingBehavior()
        {
            TextProperty.Changed.Subscribe(e =>
            {
                ((LostFocusUpdateBindingBehavior)e.Sender).OnBindingValueChanged();
            });
        }

        public static readonly StyledProperty<string> TextProperty = AvaloniaProperty.Register<LostFocusUpdateBindingBehavior, string>("Text", defaultBindingMode: BindingMode.TwoWay);

        public string Text
        {
            get => this.GetValue(TextProperty);
            set => this.SetValue(TextProperty, value);
        }

        protected override void OnAttached()
        {
            this.AssociatedObject.LostFocus += this.OnLostFocus;
            base.OnAttached();
        }

        protected override void OnDetaching()
        {
            this.AssociatedObject.LostFocus -= this.OnLostFocus;
            base.OnDetaching();
        }

        private void OnLostFocus(object? sender, RoutedEventArgs e)
        {
            if (this.AssociatedObject != null)
            {
                this.Text = this.AssociatedObject.Text;
            }
        }

        private void OnBindingValueChanged()
        {
            if (this.AssociatedObject != null)
            {
                this.AssociatedObject.Text = this.Text;
            }
        }
    }
}
